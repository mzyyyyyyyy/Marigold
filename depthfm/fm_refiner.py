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
        controlnet: Optional[ControlNetModel],
        empty_text_embed: torch.Tensor,
        ps_dropout_p: float = 0.3,
        use_controlnet: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet
        self.use_controlnet = use_controlnet

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

        # Velocity target (constant, flow matching straight path)
        v_target = z_gt - z_coarse             # (B, 4, h, w)

        # Scale t to [0, 999] for UNet timestep embedding
        t_int = (t * 999).long()

        # Text conditioning (empty)
        text_emb = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)

        if self.use_controlnet:
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
        method: str = "euler",       # "euler" or "heun"
    ) -> torch.Tensor:
        """
        Refine coarse prediction via ODE integration.

        When use_controlnet=True, uses null PS token at inference time.
        When use_controlnet=False, runs UNet only.

        method="euler": 1st-order Euler (original behaviour)
        method="heun":  2nd-order Heun (trapezoidal corrector), costs 2× NFE per step

        Returns:
            h_fine: (B, 1, H_hr, W_hr) refined height map
        """
        B = h_coarse.shape[0]
        device = h_coarse.device
        dtype = h_coarse.dtype

        z = self.encode(h_coarse)  # (B, 4, h, w)
        text_emb = self.empty_text_embed.to(device, dtype).expand(B, -1, -1)

        if self.use_controlnet:
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

        dt = 1.0 / n_steps
        ts = [i / n_steps for i in range(n_steps)]

        for t_val in ts:
            v1 = self._vel(z, t_val, control_input, text_emb)
            if method == "heun":
                t_next = min(t_val + dt, 1.0)
                z_pred = z + dt * v1
                v2 = self._vel(z_pred, t_next, control_input, text_emb)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1

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
    device: str = "cuda",
) -> FMRefiner:
    """
    Build FMRefiner from a SD2.1 pretrained checkpoint directory.

    Args:
        sd_pretrained_path: path to SD2.1 checkpoint (with unet/, vae/ subdirs)
        n_landsat_bands: number of Landsat input channels
        n_ps_bands: number of PlanetScope input channels
        ps_dropout_p: dropout probability for PS conditioning
        use_controlnet: if False, ControlNet is not built and UNet runs unconditionally
        device: device string
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    # Load VAE
    vae = AutoencoderKL.from_pretrained(sd_pretrained_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()

    # Load UNet (standard 4-channel in)
    unet = UNet2DConditionModel.from_pretrained(sd_pretrained_path, subfolder="unet")
    unet.requires_grad_(True)
    unet.train()

    if use_controlnet:
        controlnet = ControlNetModel.from_unet(unet)
        if controlnet_cond_mode == "ps_only":
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
