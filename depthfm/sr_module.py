"""SR module: SwinIR wrapper for Landsat → pseudo-PlanetScope super-resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from depthfm.swinir_arch import SwinIR


class SRModule(nn.Module):
    """
    SwinIR-based SR module.

    Expects input in [-1, 1] (Marigold convention) and produces output in [-1, 1].
    Internally converts to [0, 1] for SwinIR (img_range=1), then converts back.

    Output is bilinearly resized to target_size when specified, to handle the
    case where the SR upscale factor doesn't exactly match the LS/PS ratio.
    """

    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        upscale: int = 4,
        img_size: int = 48,
        window_size: int = 8,
        depths: list = None,
        embed_dim: int = 180,
        num_heads: list = None,
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
    ):
        super().__init__()
        depths = depths or [6, 6, 6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6, 6, 6]
        self.window_size = window_size
        self.upscale = upscale
        self.out_chans = out_chans

        self.swinir = SwinIR(
            upscale=upscale,
            in_chans=in_chans,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: (B, in_chans, H_lr, W_lr) in [-1, 1]
            target_size: (H_hr, W_hr) to resize output; if None, use native SR output
        Returns:
            (B, out_chans, H_hr, W_hr) in [-1, 1]
        """
        # [-1, 1] → [0, 1]
        x01 = (x + 1.0) / 2.0

        # Pad so spatial dims are divisible by window_size (SwinIR requirement)
        _, _, h, w = x01.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x01 = F.pad(x01, (0, pad_w, 0, pad_h), mode="reflect")

        out = self.swinir(x01)  # (B, out_chans, H_sr, W_sr)

        # Crop away the padded region (in SR space)
        out = out[:, : self.out_chans, : h * self.upscale, : w * self.upscale]

        # Resize to match PS spatial dims if they differ from SwinIR's native output
        if target_size is not None and (out.shape[2], out.shape[3]) != tuple(target_size):
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

        # [0, 1] → [-1, 1]
        return out * 2.0 - 1.0


def build_sr_module(cfg_sr: dict) -> SRModule:
    """Build SRModule from config dict."""
    return SRModule(
        in_chans=cfg_sr.get("in_chans", 3),
        out_chans=cfg_sr.get("out_chans", 3),
        upscale=cfg_sr.get("upscale", 4),
        img_size=cfg_sr.get("img_size", 48),
        window_size=cfg_sr.get("window_size", 8),
        depths=list(cfg_sr.get("depths", [6, 6, 6, 6, 6, 6])),
        embed_dim=cfg_sr.get("embed_dim", 180),
        num_heads=list(cfg_sr.get("num_heads", [6, 6, 6, 6, 6, 6])),
        mlp_ratio=float(cfg_sr.get("mlp_ratio", 2.0)),
        upsampler=cfg_sr.get("upsampler", "pixelshuffle"),
        resi_connection=cfg_sr.get("resi_connection", "1conv"),
    )
