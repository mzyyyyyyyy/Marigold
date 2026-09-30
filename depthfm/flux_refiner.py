"""
Flow Matching Refiner for Canopy Height Map, FLUX.1 backbone + LoRA.

Drop-in alternative to depthfm.fm_refiner.FMRefiner (same forward()/refine()
interface, same flow-matching training target), used by train_fm_refiner.py
when model.refiner_type == "flux".

Architecture:
  - FLUX.1 VAE (frozen, 16-ch, fp32): encodes height maps AND the conditioning
    images (Landsat, PlanetScope are 3-band RGB-like) to latents.
  - FLUX.1 transformer (bf16, frozen base + trainable LoRA on its Linear layers).
  - Conditioning is FLUX/Kontext-style *token concatenation*: Landsat (and,
    when not dropped, PlanetScope) latents are packed into tokens and appended
    to the sequence with their own position ids; only the z_t tokens' output
    is used as the velocity. No ControlNet, no new layers.
  - Text conditioning is a fixed, precomputed empty-prompt embedding (T5
    sequence truncated to `text_seq_len` tokens + CLIP pooled), so the text
    encoders are not needed at train time (see script/depth/precompute_flux_text.py).

Flow direction: this codebase's t runs coarse (t=0) -> gt (t=1):
    z_t = (1-t) z_coarse + t z_gt + (1-t) sigma eps,  v = (z_gt - z_coarse) - sigma eps
FLUX runs data (s=0) -> noise (s=1) and predicts dz/ds. With s = 1 - t the
coarse latent plays the role of the noise end, gt the data end, and
v_flux = -v. forward()/refine() hide this; the training target and sampling
match FMRefiner ("fixed" velocity parameterization only).
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Minimal LoRA (no peft dependency)
# -------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = base(x) + (alpha/rank) * B(A(x)); base is frozen, A/B are fp32 masters
    cast to the activation dtype at compute time."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out + lora * self.scale


def inject_lora(transformer: nn.Module, rank: int, alpha: float) -> int:
    """Wrap every Linear inside the double/single transformer blocks with LoRA,
    except AdaLN modulation ('norm*') linears and the input/output embedders.
    Returns the number of wrapped layers."""
    n = 0

    def _recurse(module: nn.Module, path: str):
        nonlocal n
        for name, child in list(module.named_children()):
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.Linear) and "norm" not in child_path:
                setattr(module, name, LoRALinear(child, rank, alpha))
                n += 1
            else:
                _recurse(child, child_path)

    for blocks in (transformer.transformer_blocks, transformer.single_transformer_blocks):
        _recurse(blocks, "")
    return n


# -------------------------------------------------------------------------
# Refiner
# -------------------------------------------------------------------------

def _pack(z: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, (h/2)(w/2), 4C), FLUX's 2x2 latent patchify."""
    B, C, h, w = z.shape
    z = z.view(B, C, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return z.reshape(B, (h // 2) * (w // 2), C * 4)


def _unpack(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    B, N, D = t.shape
    C = D // 4
    t = t.view(B, h // 2, w // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(B, C, h, w)


def _img_ids(h2: int, w2: int, first: int, device) -> torch.Tensor:
    """Position ids (h2*w2, 3): [stream index, row, col]. `first` separates the
    main stream (0) from the condition streams (1, 2, ...), as in Kontext."""
    ids = torch.zeros(h2, w2, 3, device=device)
    ids[..., 0] = first
    ids[..., 1] = torch.arange(h2, device=device)[:, None]
    ids[..., 2] = torch.arange(w2, device=device)[None, :]
    return ids.reshape(-1, 3)


class FluxRefiner(nn.Module):
    def __init__(
        self,
        vae,
        transformer,
        prompt_embeds: torch.Tensor,     # (1, L, 4096)
        pooled_embeds: torch.Tensor,     # (1, 768)
        ps_dropout_p: float = 0.3,
        noise_sigma: float = 0.0,
        sample_sigma: Optional[float] = None,
        sample_noise_mode: str = "sde",
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.register_buffer("prompt_embeds", prompt_embeds, persistent=False)
        self.register_buffer("pooled_embeds", pooled_embeds, persistent=False)
        self.register_buffer("txt_ids", torch.zeros(prompt_embeds.shape[1], 3), persistent=False)
        self.ps_dropout_p = ps_dropout_p
        self.noise_sigma = noise_sigma
        self.sample_sigma = noise_sigma if sample_sigma is None else sample_sigma
        if sample_noise_mode not in ("sde", "init_only"):
            raise ValueError(f"sample_noise_mode must be 'sde' or 'init_only', got {sample_noise_mode!r}")
        self.sample_noise_mode = sample_noise_mode
        self.guidance_scale = guidance_scale
        self.scale = float(vae.config.scaling_factor)
        self.shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
        self.use_controlnet = False   # interface parity with FMRefiner

    # ----- LoRA parameter helpers (used by the trainer) -----

    def lora_parameters(self) -> list:
        return [p for n, p in self.transformer.named_parameters() if "lora_" in n]

    def lora_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.transformer.named_parameters() if "lora_" in n}

    def load_lora_state_dict(self, sd: dict):
        params = dict(self.transformer.named_parameters())
        missing = [k for k in params if "lora_" in k and k not in sd]
        unexpected = [k for k in sd if k not in params]
        if missing or unexpected:
            raise RuntimeError(f"LoRA state mismatch: missing={missing[:3]}.. unexpected={unexpected[:3]}..")
        with torch.no_grad():
            for k, v in sd.items():
                params[k].copy_(v.to(params[k].device, params[k].dtype))

    @torch.no_grad()
    def broadcast_lora(self, src: int = 0):
        for p in self.lora_parameters():
            dist.broadcast(p.data, src=src)

    def all_reduce_grads(self, world_size: int):
        """Average LoRA gradients across ranks (the frozen 12B base never goes
        through DDP, so its parameters are never broadcast/synchronised)."""
        grads = [p.grad for p in self.lora_parameters()]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat /= world_size
        for g, new in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(new)

    # ----- VAE helpers -----

    @torch.no_grad()
    def _encode_rgb(self, x3: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> (B, 16, H/8, W/8), scaled, fp32."""
        z = self.vae.encode(x3.to(self.vae.dtype)).latent_dist.mode()
        return ((z - self.shift) * self.scale).float()

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Height map (B, 1, H, W) -> latent (B, 16, H/8, W/8)."""
        return self._encode_rgb(h.repeat(1, 3, 1, 1))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scale + self.shift
        out = self.vae.decode(z.to(self.vae.dtype)).sample
        return out.float().mean(dim=1, keepdim=True)

    # ----- transformer call -----

    def _cond_tokens(self, landsat_lr: torch.Tensor, ps_hr: Optional[torch.Tensor]):
        """Pack the Landsat (and optional PS) latents into tokens + position ids."""
        toks, ids = [], []
        for k, img in enumerate([landsat_lr, ps_hr]):
            if img is None:
                continue
            z = self._encode_rgb(img[:, :3])
            toks.append(_pack(z))
            ids.append(_img_ids(z.shape[2] // 2, z.shape[3] // 2, k + 1, z.device))
        return toks, ids

    def _predict(self, z: torch.Tensor, s: torch.Tensor, cond_toks: list, cond_ids: list) -> torch.Tensor:
        """FLUX-convention velocity dz/ds at latent z (B,16,h,w), noise level s in [0,1]."""
        B, C, h, w = z.shape
        main = _pack(z)
        N = main.shape[1]
        hidden = torch.cat([main] + cond_toks, dim=1).to(torch.bfloat16)
        ids = torch.cat([_img_ids(h // 2, w // 2, 0, z.device)] + cond_ids, dim=0)
        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.full((B,), self.guidance_scale, device=z.device, dtype=torch.bfloat16)
        out = self.transformer(
            hidden_states=hidden,
            encoder_hidden_states=self.prompt_embeds.expand(B, -1, -1).to(torch.bfloat16),
            pooled_projections=self.pooled_embeds.expand(B, -1).to(torch.bfloat16),
            timestep=s.to(torch.bfloat16),   # transformer scales by 1000 internally
            img_ids=ids,
            txt_ids=self.txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        return _unpack(out[:, :N].float(), h, w)

    # ----- training -----

    def forward(self, landsat_lr, ps_hr, h_coarse, h_gt) -> torch.Tensor:
        B = h_gt.shape[0]
        device = h_gt.device

        with torch.no_grad():
            z_coarse = self.encode(h_coarse)
            z_gt = self.encode(h_gt)
            # PS conditioning dropout, batch-level: with prob p the PS tokens
            # are simply absent from the sequence (that is also the inference
            # condition, replacing FMRefiner's learned null-PS token).
            drop_ps = self.training and (torch.rand(1).item() < self.ps_dropout_p)
            cond_toks, cond_ids = self._cond_tokens(landsat_lr, None if drop_ps else ps_hr)

        t = torch.rand(B, device=device)
        t4 = t.view(B, 1, 1, 1)
        z_t = (1.0 - t4) * z_coarse + t4 * z_gt
        v_target = z_gt - z_coarse
        if self.noise_sigma > 0.0:
            eps = torch.randn_like(z_coarse)
            z_t = z_t + (1.0 - t4) * self.noise_sigma * eps
            v_target = v_target - self.noise_sigma * eps

        v_flux = self._predict(z_t, 1.0 - t, cond_toks, cond_ids)   # = dz/ds = -v
        return F.mse_loss(v_flux, -v_target)

    # ----- inference -----

    def _vel(self, z, t_val: float, cond_toks, cond_ids) -> torch.Tensor:
        s = torch.full((z.shape[0],), 1.0 - t_val, device=z.device)
        return -self._predict(z, s, cond_toks, cond_ids)   # velocity in this codebase's t direction

    @torch.no_grad()
    def refine(self, landsat_lr, h_coarse, n_steps: int = 1, method: str = "euler",
               sampling_fn: str = "uniform") -> torch.Tensor:
        if sampling_fn != "uniform":
            raise ValueError("FluxRefiner supports only sampling_fn='uniform' ('fixed' velocity).")
        z = self.encode(h_coarse)
        if self.sample_sigma > 0.0 and self.sample_noise_mode == "init_only":
            z = z + self.sample_sigma * torch.randn_like(z)
        cond_toks, cond_ids = self._cond_tokens(landsat_lr, None)   # null PS = no PS tokens

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            v1 = self._vel(z, t_val, cond_toks, cond_ids)
            if method == "heun":
                z_pred = z + dt * v1
                v2 = self._vel(z_pred, min(t_val + dt, 1.0), cond_toks, cond_ids)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1
            if self.sample_sigma > 0.0 and self.sample_noise_mode == "sde" and i != n_steps - 1:
                z = z + self.sample_sigma * (dt ** 0.5) * torch.randn_like(z)
        return self.decode(z)


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

def build_flux_refiner(
    model_id: str,
    text_cache_path: str,
    lora_rank: int = 64,
    lora_alpha: float = 64.0,
    ps_dropout_p: float = 0.3,
    noise_sigma: float = 0.0,
    sample_sigma: Optional[float] = None,
    sample_noise_mode: str = "sde",
    guidance_scale: float = 1.0,
    grad_checkpointing: bool = True,
    device: str = "cpu",
) -> FluxRefiner:
    """model_id: hub id (resolved from the local HF cache, HF_HUB_OFFLINE-safe) or a
    local diffusers-format dir with vae/ and transformer/ subfolders."""
    from diffusers import AutoencoderKL, FluxTransformer2DModel

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()

    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    n_wrapped = inject_lora(transformer, lora_rank, lora_alpha)
    if grad_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()

    cache = torch.load(text_cache_path, map_location="cpu")
    refiner = FluxRefiner(
        vae=vae, transformer=transformer,
        prompt_embeds=cache["prompt_embeds"].float(), pooled_embeds=cache["pooled_embeds"].float(),
        ps_dropout_p=ps_dropout_p, noise_sigma=noise_sigma, sample_sigma=sample_sigma,
        sample_noise_mode=sample_noise_mode, guidance_scale=guidance_scale,
    )
    refiner.n_lora_layers = n_wrapped
    return refiner
