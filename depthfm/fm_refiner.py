"""
Flow Matching Refiner for Canopy Height Map.

Architecture:
  - DAv2 (frozen): Landsat → H_coarse
  - VAE (frozen):  encode/decode height maps
  - UNet (SD2.1 pretrained, trainable): predicts velocity in latent space
  - ControlNet (SD2.1 encoder clone, trainable): injects Landsat+PS conditioning

Training (Flow Matching):
  z_t = (1-t)*z_coarse + t*z_gt       # linear interpolation
  v_target = z_gt - z_coarse           # constant velocity   ("fixed", default)
           = z_gt - z_t                # InDI velocity       ("indi", see velocity_parameterization)
  v_pred = UNet(z_t, t) + ControlNet(landsat, ps_or_null)
  loss = MSE(v_pred, v_target)

  Optional InDI-style noise (noise_sigma > 0, only with velocity_parameterization=
  "fixed"): adds asymmetric noise that vanishes at the clean/z_gt end (t=1) and
  is largest at the coarse/z_coarse end (t=0) — mirrored from InDI Eq.7's
  t*eps*n (their clean end is at t=0, ours is at t=1, so the schedule flips):
    h(t) = (1-t)*noise_sigma
    z_t  = (1-t)*z_coarse + t*z_gt + h(t)*eps
    v_target = (z_gt - z_coarse) + h'(t)*eps = (z_gt - z_coarse) - noise_sigma*eps
  The h'(t)*eps correction is required so the network's target reflects this
  particular noisy sample's true marginal velocity (see sr_finer_lumi branch's
  bridge_noise for the analogous — and previously buggy, since-fixed —
  correction term for its symmetric t(1-t) schedule).

Inference (Euler integration, 1-4 steps):
  z_0 = z_coarse
  v = model(z_0, t=0, null_ps)
  "fixed": z_fine = z_0 + dt * v  (per Euler step, uniform dt)
  "indi":  z_fine = z_0 + [f(t_next)-f(t)]/(1-f(t)) * v  (per step, f = sampling_fn warp)
           — required because the InDI target is (1-t)-scaled relative to "fixed";
           dividing by (1-t) undoes that scaling to recover the true step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import AutoModelForDepthEstimation
from typing import Optional


VAE_SCALE_FACTOR = 0.18215


def _sampling_warp(x: float, kind: str) -> float:
    """Map a uniform step index in [0,1] to a warped position on the [0,1]
    flow-time axis, per CH3Depth's non-uniform sampling (Eq.8). Must pass
    through (0,0) and (1,1). 'sqrt' (concave) takes larger steps early and
    finer steps near t=1; 'square' (convex) is the opposite; 'uniform' is
    the identity (no warp)."""
    if kind == "uniform":
        return x
    elif kind == "sqrt":
        return x ** 0.5
    elif kind == "square":
        return x ** 2
    else:
        raise ValueError(f"Unknown sampling_fn: {kind!r}. Expected 'uniform', 'sqrt', or 'square'.")


class FMRefiner(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: Optional[ControlNetModel],
        empty_text_embed: torch.Tensor,
        ps_dropout_p: float = 0.3,
        use_controlnet: bool = True,
        velocity_parameterization: str = "fixed",
        noise_sigma: float = 0.0,
        sample_sigma: Optional[float] = None,
        sample_noise_mode: str = "sde",
        loss_pixel_weight: float = 0.0,
        pixel_loss_t_min: float = 0.5,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet
        self.use_controlnet = use_controlnet
        if velocity_parameterization not in ("fixed", "indi"):
            raise ValueError(
                f"velocity_parameterization must be 'fixed' or 'indi', got {velocity_parameterization!r}"
            )
        self.velocity_parameterization = velocity_parameterization

        # InDI-style asymmetric bridge noise (training) + independent SDE
        # sampling noise (inference) — see module docstring. Only supported
        # with velocity_parameterization="fixed"; the v_target correction
        # term has not been derived/implemented for "indi".
        self.noise_sigma = noise_sigma
        # sample_sigma defaults to noise_sigma (same schedule at train/sample,
        # matching InDI's own paper) but can be overridden independently —
        # e.g. sample_sigma=0.0 for deterministic ODE inference from a
        # noise-trained checkpoint without retraining.
        self.sample_sigma = noise_sigma if sample_sigma is None else sample_sigma
        # "sde": inject fresh noise at every step (Euler-Maruyama, skipped on
        #   the last step) — the existing bridge_noise-style behavior.
        # "init_only": inject sample_sigma*eps exactly once, right after
        #   encoding h_coarse, before the Euler loop starts; the loop itself
        #   is then a pure deterministic ODE with no further noise — mirrors
        #   RFMSR's inference (x = z_lr + sigma*randn(...), then plain Euler).
        if sample_noise_mode not in ("sde", "init_only"):
            raise ValueError(f"sample_noise_mode must be 'sde' or 'init_only', got {sample_noise_mode!r}")
        self.sample_noise_mode = sample_noise_mode
        if (self.noise_sigma > 0.0 or self.sample_sigma > 0.0) and velocity_parameterization == "indi":
            raise NotImplementedError(
                "noise_sigma/sample_sigma > 0 is only supported with "
                "velocity_parameterization='fixed' — the InDI-target v_target "
                "correction for noise has not been derived for 'indi'."
            )

        # Auxiliary pixel-space L1 (see forward()): weight 0 disables it; only samples with
        # t >= pixel_loss_t_min contribute (at small t the clean-latent estimate is a blurry mean).
        self.loss_pixel_weight = loss_pixel_weight
        self.pixel_loss_t_min = pixel_loss_t_min
        self.last_pixel_loss = None

        # Fixed empty text embedding (not a parameter)
        self.register_buffer("empty_text_embed", empty_text_embed)

        self.ps_dropout_p = ps_dropout_p
        # "landsat_ps": ControlNet takes cat([landsat, ps]) — default (v3/v4)
        # "ps_only":    ControlNet takes PS only (3ch), no Landsat
        self.controlnet_cond_mode = "landsat_ps"
        # Null PS token: (1, C_ps, 1, 1), broadcast over spatial dims.
        # C_ps is unknown at construction time – initialized lazily.
        self._null_ps: Optional[nn.Parameter] = None

    def _get_null_ps(self, c_ps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return null PS token of shape (1, C_ps, 1, 1), lazily initialized."""
        if self._null_ps is None:
            self._null_ps = nn.Parameter(
                torch.zeros(1, c_ps, 1, 1, device=device, dtype=dtype)
            )
        return self._null_ps.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Encode / Decode helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-channel height map to latent via VAE."""
        x3 = x.repeat(1, 3, 1, 1)              # (B,1,H,W) -> (B,3,H,W)
        h = self.vae.encoder(x3)
        moments = self.vae.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * VAE_SCALE_FACTOR

    def encode_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with grad (for training when we need gradients through decode)."""
        x3 = x.repeat(1, 3, 1, 1)
        h = self.vae.encoder(x3)
        moments = self.vae.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * VAE_SCALE_FACTOR

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to height map."""
        z = z / VAE_SCALE_FACTOR
        z = self.vae.post_quant_conv(z)
        out = self.vae.decoder(z)               # (B, 3, H, W)
        return out.mean(dim=1, keepdim=True)    # (B, 1, H, W)

    def decode_grad(self, z: torch.Tensor) -> torch.Tensor:
        """decode() with gradients flowing to z (VAE weights stay frozen)."""
        z = self.vae.post_quant_conv(z / VAE_SCALE_FACTOR)
        return self.vae.decoder(z).mean(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Forward (training step)
    # ------------------------------------------------------------------

    def forward(
        self,
        landsat_lr: torch.Tensor,   # (B, C_ls, H_hr, W_hr) – already upsampled to HR
        ps_hr: torch.Tensor,         # (B, C_ps, H_hr, W_hr)
        h_coarse: torch.Tensor,      # (B, 1, H_hr, W_hr) – DAv2 output, already at HR
        h_gt: torch.Tensor,          # (B, 1, H_hr, W_hr)
    ) -> torch.Tensor:
        """
        Compute FM training loss.

        Returns:
            Scalar MSE loss between predicted and target velocity.
        """
        B = h_gt.shape[0]
        device = h_gt.device
        dtype = h_gt.dtype

        # Encode to latent (no grad needed for targets)
        with torch.no_grad():
            z_coarse = self.encode(h_coarse)   # (B, 4, h, w)
            z_gt = self.encode(h_gt)           # (B, 4, h, w)

        # Sample t ~ Uniform(0, 1)
        t = torch.rand(B, device=device, dtype=dtype)  # (B,)

        # Linear interpolation in latent space
        t4 = t.view(B, 1, 1, 1)
        z_t = (1.0 - t4) * z_coarse + t4 * z_gt

        # Velocity target
        if self.velocity_parameterization == "indi":
            # InDI: remaining displacement to the target, shrinks as t -> 1.
            # Identically z_gt - z_t == (1-t)*(z_gt - z_coarse) (CH3Depth Eq.6/7).
            v_target = z_gt - z_t              # (B, 4, h, w)
        else:
            # Flow matching straight path: constant velocity across all t.
            v_target = z_gt - z_coarse         # (B, 4, h, w)

        # InDI-style asymmetric bridge noise (see module docstring). Only
        # active when noise_sigma > 0 (velocity_parameterization="fixed",
        # enforced in __init__).
        if self.noise_sigma > 0.0:
            eps = torch.randn_like(z_coarse)
            noise_scale = (1.0 - t4) * self.noise_sigma   # h(t): 0 at t=1 (z_gt), max at t=0 (z_coarse)
            z_t = z_t + noise_scale * eps
            v_target = v_target - self.noise_sigma * eps  # h'(t)*eps correction, h'(t) = -noise_sigma (constant)

        # Scale t to [0, 999] for UNet timestep embedding
        t_int = (t * 999).long()

        # Text conditioning (empty)
        text_emb = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)

        if self.use_controlnet:
            if self.controlnet_cond_mode == "landsat_only":
                control_input = landsat_lr                               # (B, C_ls, H, W)
            else:
                c_ps = ps_hr.shape[1]
                H_hr, W_hr = ps_hr.shape[2], ps_hr.shape[3]
                null_ps = self._get_null_ps(c_ps, device, dtype)  # (1, C_ps, 1, 1)

                # PlanetScope conditioning with dropout
                ps_cond = ps_hr.clone()
                if self.training:
                    drop_mask = torch.rand(B, device=device) < self.ps_dropout_p
                    null_spatial = null_ps.expand(1, -1, H_hr, W_hr)
                    for i in range(B):
                        if drop_mask[i]:
                            ps_cond[i] = null_spatial[0]

                if self.controlnet_cond_mode == "ps_only":
                    control_input = ps_cond                                  # (B, C_ps, H, W)
                else:
                    control_input = torch.cat([landsat_lr, ps_cond], dim=1) # (B, C_ls+C_ps, H, W)

            cn_out = self.controlnet(
                sample=z_t,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=control_input,
                return_dict=True,
            )
            unet_out = self.unet(
                sample=z_t,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=cn_out.down_block_res_samples,
                mid_block_additional_residual=cn_out.mid_block_res_sample,
                return_dict=True,
            )
        else:
            unet_out = self.unet(
                sample=z_t,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )

        v_pred = unet_out.sample                # (B, 4, h, w)
        loss = F.mse_loss(v_pred, v_target)

        # Auxiliary pixel-space L1 on the low-noise samples. z_t + (1-t)*v equals z_gt exactly when
        # v = v_target (both the noise-free and the bridge-noise parameterisations), so
        # z1_hat = z_t + (1-t)*v_pred is the network's clean-latent estimate; decode it and compare
        # with h_gt. Gradients flow through the frozen decoder into v_pred.
        self.last_pixel_loss = None
        if self.loss_pixel_weight > 0.0 and self.velocity_parameterization == "fixed":
            sel = t >= self.pixel_loss_t_min
            if sel.any():
                z1_hat = z_t[sel] + (1.0 - t4[sel]) * v_pred[sel]
                gt = h_gt[sel]
                valid = torch.isfinite(gt).float()
                err = (self.decode_grad(z1_hat) - torch.nan_to_num(gt)).abs() * valid
                pix = err.sum() / valid.sum().clamp(min=1.0)
                self.last_pixel_loss = pix.detach()
                loss = loss + self.loss_pixel_weight * pix * (sel.sum() / B)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _vel(
        self,
        z: torch.Tensor,
        t_val: float,
        control_input: Optional[torch.Tensor],
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate velocity field v(z, t) via UNet, optionally with ControlNet."""
        B = z.shape[0]
        device = z.device
        dtype = z.dtype
        t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
        t_int = (t_tensor * 999).long()
        if self.use_controlnet and control_input is not None:
            cn_out = self.controlnet(
                sample=z,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=control_input,
                return_dict=True,
            )
            unet_out = self.unet(
                sample=z,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=cn_out.down_block_res_samples,
                mid_block_additional_residual=cn_out.mid_block_res_sample,
                return_dict=True,
            )
        else:
            unet_out = self.unet(
                sample=z,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )
        return unet_out.sample

    @torch.no_grad()
    def refine(
        self,
        landsat_lr: torch.Tensor,   # (B, C_ls, H_hr, W_hr)
        h_coarse: torch.Tensor,      # (B, 1, H_hr, W_hr)
        n_steps: int = 1,
        method: str = "euler",       # "euler" or "heun" (heun only supported with velocity_parameterization="fixed")
        sampling_fn: str = "uniform",  # "uniform"/"sqrt"/"square" step warp (only used with velocity_parameterization="indi")
    ) -> torch.Tensor:
        """
        Refine coarse prediction via ODE integration.

        When use_controlnet=True, uses null PS token at inference time.
        When use_controlnet=False, runs UNet only.

        velocity_parameterization="fixed" (this model's setting, fixed at construction):
            method="euler": 1st-order Euler, uniform step size, ignores sampling_fn.
            method="heun":  2nd-order Heun (trapezoidal corrector), costs 2x NFE per step.

        velocity_parameterization="indi":
            Non-uniform step schedule via sampling_fn (CH3Depth Eq.8): the network is
            queried at the warped position f(s/n_steps), and each step's raw output is
            divided by (1 - f(s/n_steps)) to undo the (1-t) scaling baked into the InDI
            training target. Only method="euler" is implemented.

        Returns:
            h_fine: (B, 1, H_hr, W_hr) refined height map
        """
        B = h_coarse.shape[0]
        device = h_coarse.device
        dtype = h_coarse.dtype

        z = self.encode(h_coarse)  # (B, 4, h, w)
        # "init_only" mode (RFMSR-style): perturb the coarse starting point
        # once, here, before any integration — the Euler loop below then
        # runs as a pure deterministic ODE (see the per-step "sde" branch
        # further down, which is skipped entirely in this mode).
        if self.sample_sigma > 0.0 and self.sample_noise_mode == "init_only":
            z = z + self.sample_sigma * torch.randn_like(z)
        text_emb = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)

        if self.use_controlnet:
            if self.controlnet_cond_mode == "landsat_only":
                control_input = landsat_lr
            else:
                H_hr, W_hr = landsat_lr.shape[2], landsat_lr.shape[3]
                if self._null_ps is None:
                    raise RuntimeError("null_ps not initialized. Run a training forward pass first.")
                c_ps = self._null_ps.shape[1]
                null_ps = self._null_ps.expand(B, -1, H_hr, W_hr).to(device, dtype)
                if self.controlnet_cond_mode == "ps_only":
                    control_input = null_ps
                else:
                    control_input = torch.cat([landsat_lr, null_ps], dim=1)
        else:
            control_input = None

        if self.velocity_parameterization == "indi":
            if method != "euler":
                raise NotImplementedError(
                    f"refine(): method={method!r} is not supported with "
                    f"velocity_parameterization='indi' — only 'euler' is implemented."
                )
            taus = [_sampling_warp(s / n_steps, sampling_fn) for s in range(n_steps + 1)]
            for s in range(n_steps):
                t_cur, t_next = taus[s], taus[s + 1]
                v = self._vel(z, t_cur, control_input, text_emb)
                coeff = (t_next - t_cur) / (1.0 - t_cur)
                z = z + coeff * v
        else:
            if sampling_fn != "uniform":
                raise ValueError(
                    f"sampling_fn={sampling_fn!r} requires velocity_parameterization='indi' "
                    f"— a 'fixed'-parameterization model must use uniform-step Euler/Heun."
                )
            dt = 1.0 / n_steps
            ts = [i / n_steps for i in range(n_steps)]

            for i, t_val in enumerate(ts):
                is_last = (i == n_steps - 1)
                v1 = self._vel(z, t_val, control_input, text_emb)
                if method == "heun":
                    t_next = min(t_val + dt, 1.0)
                    z_pred = z + dt * v1
                    v2 = self._vel(z_pred, t_next, control_input, text_emb)
                    z = z + dt * 0.5 * (v1 + v2)
                else:
                    z = z + dt * v1
                # SDE noise injection (Euler-Maruyama), skipped on the last
                # step so z lands exactly at the clean z_gt-side endpoint
                # with no residual noise. Independent of noise_sigma — see
                # sample_sigma in __init__. Only in "sde" mode — "init_only"
                # already perturbed z once above and stays pure-ODE from here.
                if self.sample_sigma > 0.0 and self.sample_noise_mode == "sde" and not is_last:
                    z = z + self.sample_sigma * (dt ** 0.5) * torch.randn_like(z)

        h_fine = self.decode(z)
        return h_fine


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def build_fm_refiner(
    sd_pretrained_path: str,
    n_landsat_bands: int,
    n_ps_bands: int,
    ps_dropout_p: float = 0.3,
    use_controlnet: bool = True,
    controlnet_cond_mode: str = "landsat_ps",
    velocity_parameterization: str = "fixed",
    noise_sigma: float = 0.0,
    sample_sigma: Optional[float] = None,
    sample_noise_mode: str = "sde",
    device: str = "cuda",
    pretrained_unet: bool = True,
    backbone: str = "unet",
    dit_kwargs: Optional[dict] = None,
    loss_pixel_weight: float = 0.0,
    pixel_loss_t_min: float = 0.5,
) -> FMRefiner:
    """
    Build FMRefiner from a SD2.1 pretrained checkpoint directory.

    Args:
        sd_pretrained_path: path to SD2.1 checkpoint (with unet/, vae/ subdirs)
        n_landsat_bands: number of Landsat input channels
        n_ps_bands: number of PlanetScope input channels
        ps_dropout_p: dropout probability for PS conditioning
        use_controlnet: if False, ControlNet is not built and UNet runs unconditionally
        pretrained_unet: if False, the UNet is built from SD2.1's architecture config with
            RANDOM weights (ControlNet.from_unet then copies those random weights): the
            "no pretrained generative prior" control experiment. VAE and the empty-prompt
            text embedding are still taken from the SD2.1 checkpoint (frozen / constant).
        velocity_parameterization: "fixed" (constant flow-matching velocity, default)
            or "indi" (InDI-style (1-t)-scaled velocity, see module docstring)
        noise_sigma: InDI-style asymmetric bridge noise strength (training), 0.0 disables.
            Only supported with velocity_parameterization="fixed".
        sample_sigma: independent SDE noise strength at inference (refine()); defaults
            to noise_sigma when None (same schedule as training, matching the InDI
            paper), pass 0.0 explicitly for deterministic ODE inference.
        sample_noise_mode: "sde" (default; noise injected every step, bridge_noise-style)
            or "init_only" (RFMSR-style; noise injected once before integration starts,
            then a pure deterministic ODE) — see refine()/module docstring.
        backbone: "unet" (SD2.1 UNet + ControlNet, default) or "dit" (randomly initialised DiT +
            DiT-ControlNet analogue from depthfm/dit_control.py; drop-in replacements exposed as
            `.unet` / `.controlnet`, so everything else -- noise, PS dropout/null token, sampling,
            training script -- is unchanged). `dit_kwargs`: patch_size, hidden_size, depth,
            num_heads, mlp_ratio, n_control.
        device: device string
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    # Load VAE
    vae = AutoencoderKL.from_pretrained(sd_pretrained_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()

    if backbone in ("dit", "dit_concat", "dit_token", "dit_omini"):
        return _build_dit_fm_refiner(
            vae, n_landsat_bands, n_ps_bands, ps_dropout_p, use_controlnet, controlnet_cond_mode,
            velocity_parameterization, noise_sigma, sample_sigma, sample_noise_mode, dit_kwargs or {},
            variant=backbone, loss_pixel_weight=loss_pixel_weight, pixel_loss_t_min=pixel_loss_t_min)
    if backbone != "unet":
        raise ValueError(
            f"backbone must be 'unet', 'dit', 'dit_concat', 'dit_token' or 'dit_omini', got {backbone!r}")

    # Load UNet (standard 4-channel in)
    if pretrained_unet:
        unet = UNet2DConditionModel.from_pretrained(sd_pretrained_path, subfolder="unet")
    else:
        unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(sd_pretrained_path, subfolder="unet"))
        # as in the original LDM UNet (zero_module on the output conv): start by predicting
        # zero velocity instead of a random one
        nn.init.zeros_(unet.conv_out.weight)
        nn.init.zeros_(unet.conv_out.bias)
    unet.requires_grad_(True)
    unet.train()

    if use_controlnet:
        controlnet = ControlNetModel.from_unet(unet)
        if controlnet_cond_mode == "landsat_only":
            n_cond_channels = n_landsat_bands
        elif controlnet_cond_mode == "ps_only":
            n_cond_channels = n_ps_bands
        else:
            n_cond_channels = n_landsat_bands + n_ps_bands
        _adapt_controlnet_input(controlnet, n_cond_channels)
        controlnet.requires_grad_(True)
        controlnet.train()
    else:
        controlnet = None

    # Empty text embedding
    tokenizer = CLIPTokenizer.from_pretrained(sd_pretrained_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_pretrained_path, subfolder="text_encoder")
    text_encoder.eval()
    with torch.no_grad():
        tokens = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        empty_text_embed = text_encoder(tokens.input_ids)[0]  # (1, seq, D)

    model = FMRefiner(
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        empty_text_embed=empty_text_embed,
        ps_dropout_p=ps_dropout_p,
        use_controlnet=use_controlnet,
        velocity_parameterization=velocity_parameterization,
        noise_sigma=noise_sigma,
        sample_sigma=sample_sigma,
        sample_noise_mode=sample_noise_mode,
        loss_pixel_weight=loss_pixel_weight,
        pixel_loss_t_min=pixel_loss_t_min,
    )
    model.controlnet_cond_mode = controlnet_cond_mode
    return model


def _build_dit_fm_refiner(vae, n_landsat_bands, n_ps_bands, ps_dropout_p, use_controlnet, controlnet_cond_mode,
                          velocity_parameterization, noise_sigma, sample_sigma, sample_noise_mode, dit_kwargs,
                          variant="dit", loss_pixel_weight=0.0, pixel_loss_t_min=0.5):
    from depthfm.dit_control import (
        DiTBackbone, DiTConcatBackbone, DiTControl, DiTCondEncoder, DiTOminiBackbone, DiTTokenBackbone,
        DiTVAECondEncoder)
    kw = dict(patch_size=2, hidden_size=1024, depth=28, num_heads=16, mlp_ratio=4.0, n_control=14,
              cond_latent_channels=16)
    kw.update(dit_kwargs)
    n_control = kw.pop("n_control")
    cond_latent_channels = kw.pop("cond_latent_channels")
    if variant not in ("dit", "dit_omini") and not use_controlnet:
        raise ValueError(f"backbone={variant!r} needs trainer.use_controlnet: True (it enables the condition path)")
    n_cond = {"landsat_only": n_landsat_bands, "ps_only": n_ps_bands}.get(
        controlnet_cond_mode, n_landsat_bands + n_ps_bands)
    controlnet = None
    if variant == "dit":
        unet = DiTBackbone(**kw)
        if use_controlnet:
            controlnet = DiTControl(n_cond, patch_size=kw["patch_size"], hidden_size=kw["hidden_size"],
                                    n_control=n_control, num_heads=kw["num_heads"], mlp_ratio=kw["mlp_ratio"])
    elif variant == "dit_concat":
        unet = DiTConcatBackbone(cond_latent_channels=cond_latent_channels, **kw)
        controlnet = DiTCondEncoder(n_cond, "concat", patch_size=kw["patch_size"], hidden_size=kw["hidden_size"],
                                    cond_latent_channels=cond_latent_channels)
    elif variant == "dit_omini":
        # OminiControl-style: frozen-VAE-encoded [Landsat, PS] condition tokens + joint attention.
        # With use_controlnet=False there is no condition encoder and the backbone sees z_t only
        # (a plain DiT: the sequence is just the latent tokens).
        unet = DiTOminiBackbone(**kw)
        if use_controlnet:
            if controlnet_cond_mode != "landsat_ps":
                raise ValueError("backbone='dit_omini' needs trainer.controlnet_cond_mode: 'landsat_ps'")
            controlnet = DiTVAECondEncoder(vae, (n_landsat_bands, n_ps_bands), patch_size=kw["patch_size"],
                                           hidden_size=kw["hidden_size"], vae_scale=VAE_SCALE_FACTOR)
    else:
        unet = DiTTokenBackbone(**kw)
        controlnet = DiTCondEncoder(n_cond, "token", patch_size=kw["patch_size"], hidden_size=kw["hidden_size"])
    model = FMRefiner(
        vae=vae, unet=unet, controlnet=controlnet,
        empty_text_embed=torch.zeros(1, 1, 1),     # unused by the DiT (no cross-attention)
        ps_dropout_p=ps_dropout_p, use_controlnet=use_controlnet,
        velocity_parameterization=velocity_parameterization, noise_sigma=noise_sigma,
        sample_sigma=sample_sigma, sample_noise_mode=sample_noise_mode,
        loss_pixel_weight=loss_pixel_weight, pixel_loss_t_min=pixel_loss_t_min,
    )
    model.controlnet_cond_mode = controlnet_cond_mode
    return model


def _adapt_controlnet_input(controlnet: ControlNetModel, n_cond_channels: int):
    """Replace ControlNet's conv_in to accept n_cond_channels instead of 3."""
    old_conv = controlnet.controlnet_cond_embedding.conv_in
    # controlnet_cond_embedding is a ControlNetConditioningEmbedding
    # Its first conv takes the conditioning image
    old_weight = old_conv.weight  # (out, 3, kH, kW)
    out_ch = old_weight.shape[0]
    kH, kW = old_weight.shape[2], old_weight.shape[3]

    new_conv = nn.Conv2d(n_cond_channels, out_ch, kernel_size=(kH, kW),
                         padding=old_conv.padding, stride=old_conv.stride,
                         bias=old_conv.bias is not None)
    # Initialize: tile original weights across new channels
    with torch.no_grad():
        repeats = (n_cond_channels + 2) // 3  # ceil div
        tiled = old_weight.repeat(1, repeats, 1, 1)[:, :n_cond_channels, :, :]
        tiled = tiled * (3.0 / n_cond_channels)  # normalize magnitude
        new_conv.weight.copy_(tiled)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    controlnet.controlnet_cond_embedding.conv_in = new_conv


def load_dav2(
    dav2_path: str,
    backbone: str = "depth-anything/Depth-Anything-V2-Base-hf",
    out_in_scale_factor: float = 1,
) -> nn.Module:
    """
    Load a fine-tuned DepthAnythingV2Height checkpoint (.pth with model_state_dict).

    Args:
        dav2_path: path to the .pth checkpoint file
        backbone: HuggingFace model ID used when the model was trained
        out_in_scale_factor: same value used during training (usually 1 for CHM)
    """
    from depthfm.depth_anything import DepthAnythingV2Height

    model = DepthAnythingV2Height(
        backbone=backbone,
        use_geo_encoder=False,
        pretrained=False,       # weights come from the .pth file
        out_in_scale_factor=out_in_scale_factor,
    )

    ckpt = torch.load(dav2_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def run_dav2(
    dav2_model: nn.Module,
    landsat_lr: torch.Tensor,   # (B, C, H_lr, W_lr)
    target_size: tuple,          # (H_hr, W_hr)
) -> torch.Tensor:
    """
    Run DAv2 on Landsat input, upsample result to HR size.

    Returns:
        h_coarse: (B, 1, H_hr, W_hr) in normalized [-1, 1] range
    """
    # Try transformers-style forward
    try:
        out = dav2_model(pixel_values=landsat_lr)
        depth = out.predicted_depth  # (B, H, W) or (B, 1, H, W)
    except (TypeError, AttributeError):
        # Plain nn.Module that returns a tensor
        depth = dav2_model(landsat_lr)

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # (B, 1, H, W)

    # Upsample to HR resolution
    if (depth.shape[2], depth.shape[3]) != target_size:
        depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)

    return depth
