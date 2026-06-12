import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


class DepthAnythingV2Height(nn.Module):
    def __init__(self,
                 backbone="depth-anything/Depth-Anything-V2-Base-hf",
                 use_geo_encoder=False,
                 pretrained=True,
                 out_in_scale_factor=1):
        """
        out_in_scale_factor: output resolution = input resolution * out_in_scale_factor.
                             Set to 4 when labels are 4x higher resolution than input images.
        """
        super().__init__()
        self.use_geo_encoder = use_geo_encoder
        self.out_in_scale_factor = round(out_in_scale_factor)

        # Load architecture from HuggingFace; when a fine-tuned .pth is provided,
        # load_state_dict() will overwrite all weights (including upsample_head).
        self.model = AutoModelForDepthEstimation.from_pretrained(backbone)

        if self.out_in_scale_factor > 1:
            assert (self.out_in_scale_factor % 2 == 0 and
                    (self.out_in_scale_factor & (self.out_in_scale_factor - 1)) == 0), \
                "out_in_scale_factor must be a power of 2"
            num_stages = int(math.log2(self.out_in_scale_factor))
            layers = []
            in_ch = 1
            mid_ch = 64
            for i in range(num_stages):
                out_ch = mid_ch if i < num_stages - 1 else 16
                layers += [
                    nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.PixelShuffle(2),
                ]
                in_ch = out_ch
            layers.append(nn.Conv2d(in_ch, 1, kernel_size=3, padding=1))
            self.upsample_head = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, C, H, W) – first 3 channels are used as RGB input to DAv2
        returns: (B, 1, H, W)
        """
        x_rgb = x[:, :3]

        outputs = self.model(pixel_values=x_rgb)
        depth = outputs.predicted_depth.unsqueeze(1)  # (B, 1, h', w')

        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_in_scale_factor > 1:
            depth = self.upsample_head(depth)

        return depth
