import math

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


def _make_upsample_head(scale: int) -> nn.Sequential:
    """Learnable 1-channel upsampler, identical to DepthAnythingV2Height.upsample_head
    (Conv -> ReLU -> PixelShuffle(2) per x2 stage, then a final 3x3 conv)."""
    assert scale % 2 == 0 and (scale & (scale - 1)) == 0, "out_in_scale_factor must be a power of 2"
    num_stages = int(math.log2(scale))
    layers, in_ch, mid_ch = [], 1, 64
    for i in range(num_stages):
        out_ch = mid_ch if i < num_stages - 1 else 16
        layers += [
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        ]
        in_ch = out_ch
    layers.append(nn.Conv2d(in_ch, 1, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


class CHMv2Height(nn.Module):
    """Trainable CHMv2 wrapper for supervised fine-tuning (baseline_chmv2).

    Same preprocessing as run_chmv2 (dataloader [0,1] input -> per-channel
    mean/std normalization), but keeps gradients so it can be wrapped in DDP
    and optimized directly. With freeze_backbone=True (CHMv2's own recipe:
    frozen DINOv3 encoder, only the DPT head trains) the backbone is frozen
    and kept in eval mode.
    """

    PATCH = 16  # DINOv3 patch size; input H/W are padded up to a multiple of it

    def __init__(self, model_id: str, mean: list, std: list, pretrained: bool = True,
                 freeze_backbone: bool = True, upsample_input: bool = False,
                 out_in_scale_factor: int = 1):
        super().__init__()
        if pretrained:
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True)
        else:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForDepthEstimation.from_config(cfg, trust_remote_code=True)
        # Newer transformers can load checkpoints in their stored dtype (bf16/fp16); train in fp32.
        self.model.float()
        self.register_buffer("mean", torch.tensor(mean[:3], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std[:3], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.upsample_input = upsample_input
        self.out_in_scale_factor = round(out_in_scale_factor)
        assert not (upsample_input and self.out_in_scale_factor > 1), \
            "upsample_input and out_in_scale_factor > 1 would upsample twice"
        # Same trainable upsampling module as DepthAnythingV2Height (used when
        # labels are out_in_scale_factor x higher resolution than the input).
        self.upsample_head = _make_upsample_head(self.out_in_scale_factor) if self.out_in_scale_factor > 1 else None
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            backbone = getattr(self.model, "backbone", None)
            if backbone is None:
                raise AttributeError(
                    "CHMv2 model has no .backbone submodule; top-level children: "
                    f"{[n for n, _ in self.model.named_children()]}"
                )
            backbone.requires_grad_(False)
            self._backbone_ref = [backbone]  # list: avoids registering the backbone twice (duplicate state_dict keys)
            self._backbone_ref[0].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self._backbone_ref[0].eval()  # frozen encoder stays in eval mode
        return self

    def forward(self, x01: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """x01: (B, C>=3, H, W) in [0, 1]. Returns (B, 1, *target_size) in metres."""
        x = (x01[:, :3] - self.mean) / self.std.clamp(min=1e-8)
        if self.upsample_input and target_size is not None:
            x = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
        H, W = x.shape[-2:]
        pad_h, pad_w = (-H) % self.PATCH, (-W) % self.PATCH
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        depth = self.model(pixel_values=x).predicted_depth
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        # back to the (unpadded) input size, like DAv2's wrapper does
        if tuple(depth.shape[-2:]) != (H + pad_h, W + pad_w):
            depth = F.interpolate(depth, size=(H + pad_h, W + pad_w), mode="bilinear", align_corners=False)
        depth = depth[..., :H, :W]
        if self.upsample_head is not None:
            depth = self.upsample_head(depth)  # (B, 1, H*scale, W*scale)
        target_size = tuple(target_size) if target_size is not None else tuple(depth.shape[-2:])
        if tuple(depth.shape[-2:]) != target_size:
            depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)
        return depth

    @torch.no_grad()
    def forward_with_features(self, x01: torch.Tensor, target_size: tuple = None, feature_layer: int = 2):
        """forward() plus the frozen DINOv3 backbone's intermediate features, from the SAME pass.

        Returns (depth (B,1,*target_size), tokens (B, h*w, C)) where tokens are the flattened
        spatial feature map of backbone stage `feature_layer` (0..3 = DINOv3 blocks 6/12/18/24 for
        the CHMv2 ViT-L; the grid is the padded input / 16, e.g. 60x60 px -> 4x4 tokens).
        Used as the "semantic" cross-attention condition of the DiT refiner (DINOv2 features in VOSR).
        """
        store = {}
        hook = self.model.backbone.register_forward_hook(lambda m, i, o: store.__setitem__("out", o))
        try:
            depth = self.forward(x01, target_size=target_size)
        finally:
            hook.remove()
        fmap = store["out"].feature_maps[feature_layer]          # (B, C, h, w)
        return depth, fmap.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def find_unused_trainable_params(self, size: int = 64) -> list:
        """Dummy fwd/bwd; returns names of trainable params that receive no
        gradient (e.g. HF DPT-style fusion's first-layer residual conv)."""
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            dev = self.mean.device
            x = torch.rand(1, 3, size, size, device=dev)
            self(x, target_size=(size * max(self.out_in_scale_factor, 1),) * 2).sum().backward()
        names = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
        self.zero_grad(set_to_none=True)
        return names


def load_chmv2_baseline(model_dir: str, ckpt_path: str, mean: list, std: list,
                        out_in_scale_factor: int = 1) -> nn.Module:
    """Load a CHMv2Height checkpoint trained by script/depth/train_baseline_chmv2.py
    (CHMv2 + learnable upsample_head) as a frozen coarse-height predictor.

    model_dir: local dir of facebook/dinov3-vitl16-chmv2-dpt-head (only config is
        read here -- pretrained=False -- all weights come from ckpt_path).
    out_in_scale_factor: must match the value used in that training run.
    Call as model(landsat_01, target_size) -> (B, 1, *target_size) in metres.
    """
    model = CHMv2Height(model_id=model_dir, mean=mean, std=std, pretrained=False,
                        freeze_backbone=False, out_in_scale_factor=out_in_scale_factor)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
