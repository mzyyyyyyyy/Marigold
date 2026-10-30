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
                 out_in_scale_factor: int = 1, input_scale: int = 1, sampled_head: bool = False):
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
        # upsample_input resizes the input to input_scale x target size (more
        # ViT tokens); the output is resized back to target size at the end.
        self.input_scale = round(input_scale)
        assert self.input_scale == 1 or upsample_input, "input_scale > 1 needs upsample_input"
        # The head's last layers run at the input size and the result is then
        # bilinearly resized to target size; with input_scale k >= 4 (even)
        # that resize only reads 2x2 of every k x k head pixels, so
        # sampled_head evaluates those pixels alone (same output and
        # gradients, see _sampled_head_depth).
        self.sampled_head = sampled_head
        assert not sampled_head or (self.input_scale >= 4 and self.input_scale % 2 == 0), \
            "sampled_head needs an even input_scale >= 4"
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
            x = F.interpolate(x, size=tuple(s * self.input_scale for s in target_size),
                              mode="bilinear", align_corners=False)
        H, W = x.shape[-2:]
        pad_h, pad_w = (-H) % self.PATCH, (-W) % self.PATCH
        if self.sampled_head and target_size is not None:
            assert not (pad_h or pad_w), f"sampled_head needs an input size divisible by {self.PATCH}"
            return self._sampled_head_depth(x, tuple(target_size))
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

    @staticmethod
    def _upsample2x_matrix(n: int, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(len(idx), n) matrix M: M @ v = entries idx of nn.Upsample(scale_factor=2,
        mode="bilinear", align_corners=True) applied along v's length-n axis.
        Applied as a matmul rather than index_select, whose backward (an atomic
        index_add) is extremely slow on ROCm."""
        src = idx.to(torch.float64) * (n - 1) / (2 * n - 1)
        i0 = src.floor().long().clamp(max=n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        w1 = src - i0
        r = torch.arange(len(idx), device=idx.device)
        m = torch.zeros(len(idx), n, dtype=torch.float64, device=idx.device)
        m.index_put_((r, i0), 1 - w1, accumulate=True)
        m.index_put_((r, i1), w1, accumulate=True)
        return m.to(dtype)

    def _sampled_head_depth(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Same result as the default path for an input of k x target size (k =
        input_scale), computing the head's full-resolution layers only where
        they are read. Default path: fused features F at input/2 -> conv3x3 ->
        2x bilinear up -> conv3x3 -> ReLU -> conv1x1 (bin logits) -> per-pixel
        bins-to-height at the input size k*T, then a bilinear resize to T
        (align_corners=False), which for even k puts target pixel i exactly
        between head pixels k*i + k/2 - 1 and k*i + k/2: the mean of those 2x2.
        Their 3x3 conv reads rows/cols k*i + k/2 - 2 ... k*i + k/2 + 1 (inside
        the image, so the zero padding never matters): gather those 4 per
        target pixel (block i = gathered rows 4i ... 4i+3), then a valid 3x3
        conv with stride 4 at offset a in {0, 1} gives exactly head pixel
        k*i + k/2 - 1 + a. Needs the fused features at exactly input/2, which
        holds when the token grid (input / 16) is even (an odd grid makes the
        DPT fusion round up, and the default path then resizes)."""
        m, k = self.model, self.input_scale
        (Th, Tw), (H, W) = target_size, x.shape[-2:]
        assert (H, W) == (k * Th, k * Tw), f"sampled_head needs an input of {k} x the target size"
        bb = m.backbone(x)
        feats = m.head.forward_features(list(zip(bb.feature_maps, bb.cls_tokens)), H // self.PATCH, W // self.PATCH)
        assert tuple(feats.shape[-2:]) == (H // 2, W // 2), \
            f"sampled_head: fused features {tuple(feats.shape[-2:])} != input/2 (odd token grid?)"
        conv_a, up, conv_b, relu, conv_c = m.head.conv_depth.head
        assert isinstance(up, nn.Upsample) and up.scale_factor == 2 and up.align_corners
        h = conv_a(feats)  # (B, C, H/2, W/2)
        off = torch.arange(-2, 2, device=x.device)
        rows = (k * torch.arange(Th, device=x.device) + k // 2).view(-1, 1) + off  # (Th, 4)
        cols = (k * torch.arange(Tw, device=x.device) + k // 2).view(-1, 1) + off
        h = self._upsample2x_matrix(h.shape[2], rows.flatten(), h.dtype) @ h
        h = h @ self._upsample2x_matrix(h.shape[3], cols.flatten(), h.dtype).t()  # (B, C, 4*Th, 4*Tw)
        depth = 0.0
        for a in (0, 1):
            for b in (0, 1):
                o = relu(F.conv2d(h[..., a:, b:], conv_b.weight, conv_b.bias, stride=4))  # (B, C, Th, Tw)
                depth = depth + m.features_to_depth(conv_c(o))  # per-pixel: (B, 1, Th, Tw)
        return depth / 4.0

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
                        out_in_scale_factor: int = 1, upsample_input: bool = False,
                        input_scale: int = 1, sampled_head: bool = False) -> nn.Module:
    """Load a CHMv2Height checkpoint trained by script/depth/train_baseline_chmv2.py
    (CHMv2 + learnable upsample_head) as a frozen coarse-height predictor.

    model_dir: local dir of facebook/dinov3-vitl16-chmv2-dpt-head (only config is
        read here -- pretrained=False -- all weights come from ckpt_path).
    out_in_scale_factor, upsample_input, input_scale: must match the values
        used in that training run (sampled_head only changes memory/speed).
    Call as model(landsat_01, target_size) -> (B, 1, *target_size) in metres.
    """
    model = CHMv2Height(model_id=model_dir, mean=mean, std=std, pretrained=False,
                        freeze_backbone=False, out_in_scale_factor=out_in_scale_factor,
                        upsample_input=upsample_input, input_scale=input_scale,
                        sampled_head=sampled_head)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
