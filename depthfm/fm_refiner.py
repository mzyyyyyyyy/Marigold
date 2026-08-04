"""
Flow Matching Refiner for Canopy Height Map.

Architecture:
  - DAv2 (frozen): Landsat → H_coarse
  - VAE (frozen):  encode/decode height maps
  - UNet (SD2.1 pretrained, trainable): predicts velocity in latent space
  - ControlNet (SD2.1 encoder clone, trainable): injects Landsat+PS conditioning

Training (Flow Matching):
  z_t = (1-t)*z_coarse + t*z_gt       # linear interpolation
  v_target = z_gt - z_coarse           # constant velocity
  v_pred = UNet(z_t, t) + ControlNet(landsat, ps_or_null)
  loss = MSE(v_pred, v_target)

Inference (Euler integration, 1-4 steps):
  z_0 = z_coarse
  v = model(z_0, t=0, null_ps)
  z_fine = z_0 + dt * v  (per Euler step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import AutoModelForDepthEstimation
from typing import Optional


VAE_SCALE_FACTOR = 0.18215


class FMRefiner(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        empty_text_embed: torch.Tensor,
        ps_dropout_p: float = 0.3,
        bridge_sigma: float = 0.0,
        concat_z_coarse: bool = False,
        refine_threshold: float = 0.0,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet

        # Fixed empty text embedding (not a parameter)
        self.register_buffer("empty_text_embed", empty_text_embed)

        self.ps_dropout_p = ps_dropout_p
        self.bridge_sigma = bridge_sigma
        self.concat_z_coarse = concat_z_coarse
        # Pixel-space threshold for selective refinement during training.
        # 0.0 = disabled (all pixels refined normally).
        self.refine_threshold = refine_threshold
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

    # ------------------------------------------------------------------
    # Forward (training step)
    # ------------------------------------------------------------------

    def forward(
        self,
        h_coarse: torch.Tensor,                          # (B, 1, H_hr, W_hr)
        h_gt: torch.Tensor,                              # (B, 1, H_hr, W_hr)
        ps_hr: Optional[torch.Tensor] = None,            # (B, C_ps, H_hr, W_hr) or None
        ps_cond_override: Optional[torch.Tensor] = None, # bypass internal dropout
        # legacy arg kept for call-site compatibility; ignored
        landsat_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute FM training loss.

        When ps_hr is None (and ps_cond_override is None), runs UNet-only (no ControlNet).
        When ps_hr is provided, ControlNet uses PS-only conditioning (no Landsat).

        Returns:
            Scalar MSE loss between predicted and target velocity.
        """
        B = h_gt.shape[0]
        device = h_gt.device
        dtype = h_gt.dtype

        # Encode to latent
        with torch.no_grad():
            z_coarse = self.encode(h_coarse)
            z_gt     = self.encode(h_gt)

        t  = torch.rand(B, device=device, dtype=dtype)
        t4 = t.view(B, 1, 1, 1)

        # Selective refinement mask: pixels where |h_gt - h_coarse| > threshold
        # are trained normally; others are forced to keep coarse (v_target=0).
        # Mask is computed in pixel space then downsampled to latent resolution (÷8).
        if self.refine_threshold > 0.0:
            with torch.no_grad():
                mask_pixel = ((h_gt - h_coarse).abs() > self.refine_threshold).float()
                # Downsample to latent spatial resolution
                lat_h, lat_w = z_coarse.shape[2], z_coarse.shape[3]
                mask_lat = F.interpolate(mask_pixel, size=(lat_h, lat_w), mode="nearest")
                mask_lat = mask_lat.expand_as(z_coarse)
        else:
            mask_lat = None

        # Bridge Matching: add noise proportional to sqrt(t*(1-t)) at midpoint.
        if self.bridge_sigma > 0.0:
            eps = torch.randn_like(z_coarse)
            noise_scale = self.bridge_sigma * torch.sqrt(t4 * (1.0 - t4))
            z_t = (1.0 - t4) * z_coarse + t4 * z_gt + noise_scale * eps
        else:
            z_t = (1.0 - t4) * z_coarse + t4 * z_gt

        # Selective refinement: for easy pixels (mask=0), fix z_t = z_coarse and v_target = 0.
        # Both input and target must be consistent: model sees z_coarse and learns to stay there.
        if mask_lat is not None:
            z_t      = z_t * mask_lat + z_coarse * (1.0 - mask_lat)
            v_target = (z_gt - z_coarse) * mask_lat
        else:
            v_target = z_gt - z_coarse

        text_emb = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)
        t_int    = (t * 999).long()

        unet_sample = torch.cat([z_t, z_coarse], dim=1) if self.concat_z_coarse else z_t

        # Build control_input (PS only) or None
        if ps_cond_override is not None:
            control_input = ps_cond_override
        elif ps_hr is not None:
            H_hr, W_hr = ps_hr.shape[2], ps_hr.shape[3]
            c_ps    = ps_hr.shape[1]
            null_ps = self._get_null_ps(c_ps, device, dtype)
            ps_cond = ps_hr.clone()
            if self.training:
                drop_mask    = torch.rand(B, device=device) < self.ps_dropout_p
                null_spatial = null_ps.expand(1, -1, H_hr, W_hr)
                for i in range(B):
                    if drop_mask[i]:
                        ps_cond[i] = null_spatial[0]
            control_input = ps_cond
        else:
            control_input = None

        if control_input is not None:
            cn_out = self.controlnet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=control_input, return_dict=True,
            )
            unet_out = self.unet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=cn_out.down_block_res_samples,
                mid_block_additional_residual=cn_out.mid_block_res_sample,
                return_dict=True,
            )
        else:
            unet_out = self.unet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )

        v_pred = unet_out.sample
        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _vel(
        self,
        z: torch.Tensor,
        z_coarse: torch.Tensor,
        t_val: float,
        control_input: Optional[torch.Tensor],
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate velocity field v(z, t). Uses ControlNet iff control_input is not None."""
        B = z.shape[0]
        device = z.device
        dtype = z.dtype
        t_int = (torch.full((B,), t_val, device=device, dtype=dtype) * 999).long()
        unet_sample = torch.cat([z, z_coarse], dim=1) if self.concat_z_coarse else z
        if control_input is not None:
            cn_out = self.controlnet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=control_input, return_dict=True,
            )
            unet_out = self.unet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=cn_out.down_block_res_samples,
                mid_block_additional_residual=cn_out.mid_block_res_sample,
                return_dict=True,
            )
        else:
            unet_out = self.unet(
                sample=unet_sample, timestep=t_int,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )
        return unet_out.sample

    @torch.no_grad()
    def refine(
        self,
        h_coarse: torch.Tensor,      # (B, 1, H_hr, W_hr)
        n_steps: int = 1,
        method: str = "euler",
        ps_hr: Optional[torch.Tensor] = None,  # (B, C_ps, H_hr, W_hr) or None
        # legacy arg; ignored
        landsat_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Refine coarse prediction via ODE integration.

        ps_hr=None  → UNet-only, no ControlNet (unconditional).
        ps_hr given → ControlNet conditioned on PS only.
        """
        B      = h_coarse.shape[0]
        device = h_coarse.device
        dtype  = h_coarse.dtype

        z_coarse_lat = self.encode(h_coarse)
        z            = z_coarse_lat.clone()
        text_emb     = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)

        if ps_hr is not None:
            control_input = ps_hr.to(device, dtype)
        elif self._null_ps is not None:
            H_hr, W_hr    = h_coarse.shape[2], h_coarse.shape[3]
            control_input = self._null_ps.expand(B, -1, H_hr, W_hr).to(device, dtype)
        else:
            control_input = None

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val      = i / n_steps
            is_last    = (i == n_steps - 1)
            v1 = self._vel(z, z_coarse_lat, t_val, control_input, text_emb)
            if method == "heun":
                t_next = min(t_val + dt, 1.0)
                v2 = self._vel(z + dt * v1, z_coarse_lat, t_next, control_input, text_emb)
                z  = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1
            # SDE noise injection (skip on last step to avoid noise at t=1)
            if self.bridge_sigma > 0.0 and not is_last:
                z = z + self.bridge_sigma * (dt ** 0.5) * torch.randn_like(z)

        return self.decode(z)

    @torch.no_grad()
    def refine_with_ps(
        self,
        ps_hr: torch.Tensor,         # (B, C_ps, H_hr, W_hr)
        h_coarse: torch.Tensor,      # (B, 1, H_hr, W_hr)
        n_steps: int = 1,
        method: str = "euler",
        # legacy arg; ignored
        landsat_lr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine using PS-only ControlNet conditioning."""
        return self.refine(h_coarse, n_steps=n_steps, method=method, ps_hr=ps_hr)


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def _adapt_unet_input(unet: UNet2DConditionModel, in_channels: int = 8):
    """Expand UNet conv_in from 4 to in_channels. New channels are zero-initialized."""
    old = unet.conv_in
    new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, padding=old.padding)
    with torch.no_grad():
        new.weight[:, :old.in_channels] = old.weight
        new.weight[:, old.in_channels:] = 0.0
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.config.in_channels = in_channels


def build_fm_refiner(
    sd_pretrained_path: str,
    n_ps_bands: int,
    ps_dropout_p: float = 0.3,
    bridge_sigma: float = 0.0,
    concat_z_coarse: bool = False,
    refine_threshold: float = 0.0,
    device: str = "cuda",
    # legacy arg; kept for call-site compatibility, ignored
    n_landsat_bands: int = 0,
) -> FMRefiner:
    """
    Build FMRefiner from a SD2.1 pretrained checkpoint directory.

    Args:
        sd_pretrained_path: path to SD2.1 checkpoint (with unet/, vae/ subdirs)
        n_landsat_bands: number of Landsat input channels
        n_ps_bands: number of PlanetScope input channels
        ps_dropout_p: dropout probability for PS conditioning
        device: device string
    """
    import os
    from transformers import CLIPTextModel, CLIPTokenizer

    # Load VAE
    vae = AutoencoderKL.from_pretrained(sd_pretrained_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(sd_pretrained_path, subfolder="unet")
    if concat_z_coarse:
        _adapt_unet_input(unet, in_channels=8)
    unet.requires_grad_(True)
    unet.train()

    # Build ControlNet from UNet encoder (conditioned on PS only)
    controlnet = ControlNetModel.from_unet(unet)
    _adapt_controlnet_input(controlnet, n_ps_bands)
    controlnet.requires_grad_(True)
    controlnet.train()

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
        bridge_sigma=bridge_sigma,
        concat_z_coarse=concat_z_coarse,
        refine_threshold=refine_threshold,
    )
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
