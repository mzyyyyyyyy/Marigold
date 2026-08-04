import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


def load_chmv2(model_id: str = "facebook/dinov3-vitl16-chmv2-dpt-head") -> nn.Module:
    model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def run_chmv2(
    model: nn.Module,
    landsat_lr: torch.Tensor,   # (B, C, H_lr, W_lr), in [-1, 1]
    target_size: tuple,
    mean: list,
    std: list,
) -> torch.Tensor:
    x = (landsat_lr + 1.0) / 2.0
    n_ch = x.shape[1]
    mean_t = torch.tensor(mean[:n_ch], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t  = torch.tensor(std[:n_ch],  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x = (x - mean_t) / std_t.clamp(min=1e-8)

    out = model(pixel_values=x[:, :3])
    depth = out.predicted_depth
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if (depth.shape[2], depth.shape[3]) != target_size:
        depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)
    return depth
