"""
Residual flow-matching refiner with a (VOSR-pretrained) LightningDiT backbone.

Same forward()/refine() interface and the same flow formulation as FMRefiner / FluxRefiner
(RFMSR-style residual flow: the source of the flow is the coarse latent + sigma*eps, the
target is the ground-truth latent; this codebase's t runs coarse(0) -> gt(1), the DiT's
timestep runs s = 1 - t, v_dit = -v), but with a small, fully trained DiT instead of a SD2.1
UNet + ControlNet or a LoRA-tuned FLUX:

  * backbone: LightningDiT-0.5B (dim 1024, depth 28, 16 heads, patch 2, QK-norm, SwiGLU, RoPE,
    RMSNorm), initialised from the public VOSR_0.5B_ms checkpoint (vision-only SR model trained
    from scratch; also the initialisation of RFMSR), then fine-tuned with all weights.
  * latent space: SD2.1 VAE (frozen), identical to the one VOSR was trained with.
  * input = channel concat [coarse latent y0 (VOSR's "LQ" slot), state z_s, (Landsat latent),
    PlanetScope latent]: the first 8 input channels of the patch embedding are the pretrained
    ones, the Landsat/PS channels start at zero. The Landsat latent is the VAE encoding of the
    Landsat patch bilinearly upsampled to the target size (what the SD UNet's ControlNet received);
    PS is dropped (zeros) for a fraction of the training samples and is always absent (zeros) at
    inference, as in the other refiners.
  * "semantic" condition (DINOv2 features in VOSR), selectable (semantic_mode):
      - "chmv2": tokens from the frozen CHMv2 / DINOv3 backbone for the Landsat patch, fed through
        the DiT's cross-attention (the 768-d input projection is re-initialised for 1024-d tokens);
      - "zeros": a single all-zero token with the ORIGINAL 768-d pretrained path left untouched --
        VOSR's own "null" semantic condition (10% of its training samples), so nothing pretrained
        is re-initialised (all-identical keys make the cross-attention output independent of the
        number of zero tokens).
  * EMA copy of the DiT is what refine() (validation / inference) uses.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthfm.lightningdit import LightningDiT


class DiTRefiner(nn.Module):
    def __init__(
        self,
        vae,
        dit: LightningDiT,
        ema: Optional[LightningDiT],
        ps_dropout_p: float = 0.3,
        noise_sigma: float = 0.0,
        sample_sigma: Optional[float] = None,
        sample_noise_mode: str = "init_only",
        ema_decay: float = 0.998,
        use_landsat: bool = False,
        semantic_mode: str = "chmv2",
        z_dims: int = 1024,
    ):
        super().__init__()
        if semantic_mode not in ("chmv2", "zeros"):
            raise ValueError(f"semantic_mode must be 'chmv2' or 'zeros', got {semantic_mode!r}")
        self.use_landsat = use_landsat
        self.semantic_mode = semantic_mode
        self.z_dims = z_dims
        self.vae = vae
        self.dit = dit
        self.ema = ema
        self.ps_dropout_p = ps_dropout_p
        self.noise_sigma = noise_sigma
        self.sample_sigma = noise_sigma if sample_sigma is None else sample_sigma
        if sample_noise_mode not in ("sde", "init_only"):
            raise ValueError(f"sample_noise_mode must be 'sde' or 'init_only', got {sample_noise_mode!r}")
        self.sample_noise_mode = sample_noise_mode
        self.ema_decay = ema_decay
        self.ema_updates = 0
        self.scale = float(vae.config.scaling_factor)
        self.use_controlnet = False   # interface parity with FMRefiner

    # ----- VAE helpers (same conventions as FMRefiner) -----

    @torch.no_grad()
    def _encode_rgb(self, x3: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x3.to(self.vae.dtype)).latent_dist.mode().float() * self.scale

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Height map (B, 1, H, W) in [-1, 1] -> latent (B, 4, H/8, W/8)."""
        return self._encode_rgb(h.repeat(1, 3, 1, 1))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode((z / self.scale).to(self.vae.dtype)).sample
        return out.float().mean(dim=1, keepdim=True)

    # ----- EMA -----

    @torch.no_grad()
    def update_ema(self):
        """EMA of the DiT weights (with the usual warm-up so early steps are not dominated by the init)."""
        self.ema_updates += 1
        d = min(self.ema_decay, (1.0 + self.ema_updates) / (10.0 + self.ema_updates))
        src = self.dit.module if hasattr(self.dit, "module") else self.dit   # unwrap DDP
        for pe, p in zip(self.ema.parameters(), src.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _dit_inp(self, y0, z, ls_lat, ps_lat):
        parts = [y0, z] + ([ls_lat] if self.use_landsat else []) + [ps_lat]
        return torch.cat(parts, dim=1)

    def _semantic(self, coarse_feats, B, device):
        """Cross-attention tokens: CHMv2 backbone tokens, or one zero token (VOSR's null condition)."""
        if self.semantic_mode == "chmv2":
            assert coarse_feats is not None, "semantic_mode='chmv2' needs coarse_feats (CHMv2 backbone tokens)"
            return [coarse_feats]
        return [torch.zeros(B, 1, self.z_dims, device=device)]

    def _ac(self, device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    # ----- training -----

    def forward(self, landsat_lr, ps_hr, h_coarse, h_gt, landsat_native=None, coarse_feats=None) -> torch.Tensor:
        B = h_gt.shape[0]
        device = h_gt.device
        with torch.no_grad():
            z_coarse = self.encode(h_coarse)
            z_gt = self.encode(h_gt)
            ls_lat = self._encode_rgb(landsat_lr[:, :3]) if self.use_landsat else None
            ps_lat = self._encode_rgb(ps_hr[:, :3])
            if self.training and self.ps_dropout_p > 0:
                drop = torch.rand(B, device=device) < self.ps_dropout_p
                ps_lat = ps_lat * (~drop).float().view(B, 1, 1, 1)   # per-sample null PS = zeros

        t = torch.rand(B, device=device)
        t4 = t.view(B, 1, 1, 1)
        z_t = (1.0 - t4) * z_coarse + t4 * z_gt
        v_target = z_gt - z_coarse
        if self.noise_sigma > 0.0:
            eps = torch.randn_like(z_coarse)
            z_t = z_t + (1.0 - t4) * self.noise_sigma * eps
            v_target = v_target - self.noise_sigma * eps

        with self._ac(device):
            v_dit = self.dit(self._dit_inp(z_coarse, z_t, ls_lat, ps_lat), 1.0 - t,
                             self._semantic(coarse_feats, B, device))                          # = dz/ds = -v
        return F.mse_loss(v_dit.float(), -v_target)

    # ----- inference -----

    @torch.no_grad()
    def refine(self, landsat_lr, h_coarse, n_steps: int = 1, method: str = "euler",
               sampling_fn: str = "uniform", landsat_native=None, coarse_feats=None) -> torch.Tensor:
        if sampling_fn != "uniform":
            raise ValueError("DiTRefiner supports only sampling_fn='uniform'.")
        model = self.ema if self.ema is not None else self.dit
        y0 = self.encode(h_coarse)
        ls_lat = self._encode_rgb(landsat_lr[:, :3]) if self.use_landsat else None
        sem = self._semantic(coarse_feats, y0.shape[0], h_coarse.device)
        z = y0.clone()
        if self.sample_sigma > 0.0 and self.sample_noise_mode == "init_only":
            z = z + self.sample_sigma * torch.randn_like(z)
        ps_lat = torch.zeros_like(y0)   # null PS
        device = h_coarse.device

        def vel(zz, t_val):
            s = torch.full((zz.shape[0],), 1.0 - t_val, device=device)
            with self._ac(device):
                return -model(self._dit_inp(y0, zz, ls_lat, ps_lat), s, sem).float()

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            v1 = vel(z, t_val)
            if method == "heun":
                v2 = vel(z + dt * v1, min(t_val + dt, 1.0))
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1
            if self.sample_sigma > 0.0 and self.sample_noise_mode == "sde" and i != n_steps - 1:
                z = z + self.sample_sigma * (dt ** 0.5) * torch.randn_like(z)
        return self.decode(z)


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

def load_vosr_pretrained(dit: LightningDiT, ckpt_path: str) -> dict:
    """Load a VOSR checkpoint into `dit`, tolerating the deliberate shape differences:
      * x_embedder.proj.weight: VOSR has 8 input channels [LQ latent, noisy latent]; extra input
        channels (here: PlanetScope) are zero-initialised so the model starts as the pretrained one.
      * layer_norm / mlp_ca.fc1.weight: shaped by the semantic-feature width (768 for DINOv2-B);
        re-initialised when z_dims differs (1024 for CHMv2 features).
    Returns a small report dict."""
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    own = dit.state_dict()
    new, reinit, skipped = {}, [], []
    for k, v in sd.items():
        if k not in own:
            skipped.append(k)
        elif k == "x_embedder.proj.weight" and v.shape != own[k].shape:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            new[k] = w
        elif v.shape == own[k].shape:
            new[k] = v
        else:
            reinit.append(k)
    dit.load_state_dict(new, strict=False)
    return {"loaded": len(new), "reinitialised": reinit, "skipped": skipped,
            "missing": [k for k in own if k not in new]}


def build_dit_refiner(
    sd_pretrained_path: str,
    vosr_ckpt: Optional[str],
    z_dims: int = 1024,
    ps_dropout_p: float = 0.3,
    noise_sigma: float = 1.0,
    sample_sigma: Optional[float] = None,
    sample_noise_mode: str = "init_only",
    ema_decay: float = 0.998,
    use_landsat: bool = False,
    semantic: str = "chmv2",
    grad_checkpointing: bool = False,
    dim: int = 1024,
    depth: int = 28,
    num_heads: int = 16,
):
    """VOSR-0.5B architecture by default. vosr_ckpt=None -> random initialisation."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(sd_pretrained_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    # channels: y0 (4) + state (4) + [Landsat (4)] + PS (4); the first 8 are VOSR's pretrained ones
    dit = LightningDiT(input_size=64, patch_size=2, in_channels=12 + (4 if use_landsat else 0), out_channels=4,
                       hidden_size=dim,
                       depth=depth, num_heads=num_heads, mlp_ratio=4, z_dims=z_dims, encdim_ratio=3,
                       use_checkpoint=grad_checkpointing)
    report = load_vosr_pretrained(dit, vosr_ckpt) if vosr_ckpt else None
    ema = copy.deepcopy(dit).eval().requires_grad_(False)
    model = DiTRefiner(vae, dit, ema, ps_dropout_p=ps_dropout_p, noise_sigma=noise_sigma,
                       sample_sigma=sample_sigma, sample_noise_mode=sample_noise_mode, ema_decay=ema_decay,
                       use_landsat=use_landsat, semantic_mode=semantic, z_dims=z_dims)
    model.load_report = report
    return model
