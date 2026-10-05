"""
DiT replacement for the SD2.1 UNet + ControlNet pair used by FMRefiner (fm_refiner-R-lumi-5).

Only the denoising backbone changes; everything FMRefiner does around it (VAE latent space, flow /
noise schedule, PS dropout + null token, Euler sampling, ...) is untouched. To keep it that way the
two modules below are drop-ins for `UNet2DConditionModel` / `ControlNetModel` as FMRefiner calls them:

  DiTControl(sample, timestep, encoder_hidden_states, controlnet_cond, return_dict)
        -> out.down_block_res_samples (tuple of (B, N, D) token residuals), out.mid_block_res_sample (None)
  DiTBackbone(sample, timestep, encoder_hidden_states, down_block_additional_residuals,
              mid_block_additional_residual, return_dict) -> out.sample (B, 4, H, W)

  * input of the backbone is z_t only (4 channels), exactly like the UNet; the (constant, empty) text
    embedding is ignored, so there is no cross-attention;
  * `timestep` is the same integer 0..999 FMRefiner feeds the UNet (t*999), embedded sinusoidally;
  * ControlNet analogue (PixArt-delta style): a separate trainable branch of the first `n_control`
    blocks sees (z_t, pixel-space condition [Landsat, PS-or-null]); the condition goes through a conv
    stack (8x down to latent resolution, then patchify) whose last layer is zero-initialised, and
    every control block's output goes through a zero-initialised linear and is ADDED to the output of
    the matching backbone block. As in ControlNet, the branch starts as a no-op;
  * the final layer is zero-initialised (as the UNet's conv_out in the random-init control), so
    the model starts out predicting zero velocity;
  * RoPE positions are ABSOLUTE token indices (no rescaling to a reference grid): the training patches
    come in two sizes (240 / 480 px), and a UNet is translation-invariant at a fixed pixel scale, so
    "one token apart" must mean the same ground distance at both sizes.

Block internals (adaLN, QK-norm RoPE attention, SwiGLU, RMSNorm) are LightningDiT's (depthfm/lightningdit.py).
"""

import torch
import torch.nn as nn

from depthfm.lightningdit import (
    FinalLayer, LightningDiTBlock, PatchEmbed, TimestepEmbedder, _rotate_half,
)


class _Out(dict):
    """dict with attribute access (diffusers-output look-alike, DDP-friendly)."""
    __getattr__ = dict.__getitem__


class AbsRoPE2D(nn.Module):
    """2D RoPE (EVA-02 layout as in lightningdit.VisionRotaryEmbeddingFast) with absolute positions."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim, self.theta = head_dim // 2, theta

    def _tables(self, grid: int, device):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device)[: self.dim // 2].float() / self.dim))
        pos = torch.arange(grid, device=device).float()
        f = (pos[:, None] * freqs[None]).repeat_interleave(2, dim=-1)
        f = torch.cat([f[:, None, :].expand(grid, grid, -1), f[None, :, :].expand(grid, grid, -1)], dim=-1)
        f = f.reshape(grid * grid, -1)
        return f.cos(), f.sin()

    def forward(self, t, grid: int):
        cos, sin = self._tables(grid, t.device)
        return t * cos.to(t.dtype) + _rotate_half(t) * sin.to(t.dtype)


def _zero_linear(dim: int) -> nn.Linear:
    lin = nn.Linear(dim, dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _init_blocks(module: nn.Module):
    def _basic(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    module.apply(_basic)


class _TimeAndTokens(nn.Module):
    """patch embedding + timestep conditioning shared by the backbone and the control branch."""

    def __init__(self, in_channels, patch_size, hidden_size):
        super().__init__()
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def init(self):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

    def forward(self, x, timestep):
        tok = self.x_embedder(x)
        c = self.t_embedder(timestep.float())
        return tok, c, self.t_block(c)


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class DiTBackbone(nn.Module):
    """Drop-in for UNet2DConditionModel (4-ch latent in / 4-ch velocity out)."""

    def __init__(self, patch_size=2, in_channels=4, out_channels=4, hidden_size=1024, depth=28,
                 num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_size, self.out_channels, self.depth, self.hidden_size = patch_size, out_channels, depth, hidden_size
        self.embed = _TimeAndTokens(in_channels, patch_size, hidden_size)
        self.rope = AbsRoPE2D(hidden_size // num_heads)
        self.blocks = nn.ModuleList([LightningDiTBlock(hidden_size, num_heads, mlp_ratio, z_dims=None)
                                     for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        _init_blocks(self)
        self.embed.init()
        for p in (self.final_layer.adaLN_modulation[-1].weight, self.final_layer.adaLN_modulation[-1].bias,
                  self.final_layer.linear.weight, self.final_layer.linear.bias):
            nn.init.zeros_(p)

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, h * p, h * p)

    def forward(self, sample, timestep, encoder_hidden_states=None, down_block_additional_residuals=None,
                mid_block_additional_residual=None, return_dict=True):
        B, _, H, W = sample.shape
        assert H == W and H % self.patch_size == 0, "square latents with side divisible by patch_size only"
        grid = H // self.patch_size
        rope = lambda q: self.rope(q, grid)
        res = down_block_additional_residuals
        with _autocast(sample.device):
            x, c, c0 = self.embed(sample, timestep)
            for i, blk in enumerate(self.blocks):
                x = blk(x, c0, None, rope)
                if res is not None and i < len(res):
                    x = x + res[i]
            out = self.unpatchify(self.final_layer(x, c)).float()
        return _Out(sample=out)


class DiTControl(nn.Module):
    """Drop-in for ControlNetModel: first `n_control` blocks, zero-linear outputs, zero-init conditioning."""

    def __init__(self, cond_channels, patch_size=2, in_channels=4, hidden_size=1024, n_control=14,
                 num_heads=16, mlp_ratio=4.0, cond_hidden=(32, 64, 128, 256)):
        super().__init__()
        self.patch_size, self.n_control = patch_size, n_control
        self.embed = _TimeAndTokens(in_channels, patch_size, hidden_size)
        self.rope = AbsRoPE2D(hidden_size // num_heads)
        # pixel-space condition -> latent resolution (8x down, like ControlNetConditioningEmbedding)
        # -> tokens (patchify with the backbone's patch size); last layer zero-initialised
        chs = (cond_channels,) + tuple(cond_hidden)
        layers = [nn.Conv2d(chs[0], chs[1], 3, padding=1), nn.SiLU()]
        for a, b in zip(chs[1:-1], chs[2:]):
            layers += [nn.Conv2d(a, a, 3, padding=1), nn.SiLU(), nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU()]
        layers += [nn.Conv2d(chs[-1], chs[-1], 3, padding=1), nn.SiLU()]
        self.cond_embed = nn.Sequential(*layers)
        self.cond_patch = nn.Conv2d(chs[-1], hidden_size, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([LightningDiTBlock(hidden_size, num_heads, mlp_ratio, z_dims=None)
                                     for _ in range(n_control)])
        self.zero_linears = nn.ModuleList([_zero_linear(hidden_size) for _ in range(n_control)])
        _init_blocks(self)
        self.embed.init()
        for m in self.cond_embed:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cond_patch.weight)
        nn.init.zeros_(self.cond_patch.bias)
        for zl in self.zero_linears:        # _init_blocks re-initialised them
            nn.init.zeros_(zl.weight)
            nn.init.zeros_(zl.bias)

    def forward(self, sample, timestep, encoder_hidden_states=None, controlnet_cond=None, return_dict=True):
        B, _, H, W = sample.shape
        grid = H // self.patch_size
        rope = lambda q: self.rope(q, grid)
        with _autocast(sample.device):
            x, _, c0 = self.embed(sample, timestep)
            cond = self.cond_patch(self.cond_embed(controlnet_cond)).flatten(2).transpose(1, 2)
            assert cond.shape == x.shape, f"condition tokens {tuple(cond.shape)} != latent tokens {tuple(x.shape)}"
            x = x + cond
            outs = []
            for blk, zl in zip(self.blocks, self.zero_linears):
                x = blk(x, c0, None, rope)
                outs.append(zl(x).float())
        return _Out(down_block_res_samples=tuple(outs), mid_block_res_sample=None)


# ---------------------------------------------------------------------------------------------
# fm_refiner-R-lumi-12 / -13: single-backbone DiTs, no control branch.
#
# Both reuse FMRefiner's UNet/ControlNet call interface, with the condition ENCODER sitting where
# the ControlNet sits: `controlnet(sample, timestep, enc, controlnet_cond)` returns the encoded
# condition as `down_block_res_samples=(cond,)`, and the backbone receives it through
# `down_block_additional_residuals=(cond,)`. The pixel-space condition [Landsat, PS-or-null]
# (PS dropout / null token handled by FMRefiner, unchanged) goes through the same conv stack as
# DiTControl.cond_embed (8x down to latent resolution), then:
#   dit_concat : 1x1 conv -> `cond_latent_channels` channels, CONCATENATED to z_t along channels
#                before the patch embedding (in_channels = 4 + cond_latent_channels);
#   dit_token  : patchify conv -> (B, N, D) condition tokens, CONCATENATED to the latent tokens
#                along the SEQUENCE (OminiControl style): joint full self-attention over [X; C],
#                the condition tokens reuse the 2D RoPE positions of the latent tokens they are
#                aligned with, only the X tokens go through the output head.
# Everything else (blocks, adaLN time conditioning, zero-init output layer, absolute RoPE, bf16
# autocast, random init) is the same as DiTBackbone.
# ---------------------------------------------------------------------------------------------

def _cond_conv_stack(cond_channels, cond_hidden=(32, 64, 128, 256)):
    """pixel condition -> latent resolution (8x down); same layers as DiTControl.cond_embed."""
    chs = (cond_channels,) + tuple(cond_hidden)
    layers = [nn.Conv2d(chs[0], chs[1], 3, padding=1), nn.SiLU()]
    for a, b in zip(chs[1:-1], chs[2:]):
        layers += [nn.Conv2d(a, a, 3, padding=1), nn.SiLU(), nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU()]
    layers += [nn.Conv2d(chs[-1], chs[-1], 3, padding=1), nn.SiLU()]
    stack = nn.Sequential(*layers)
    for m in stack:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return stack, chs[-1]


class DiTCondEncoder(nn.Module):
    """Stands in for ControlNetModel in the -12 / -13 variants (see block comment above)."""

    def __init__(self, cond_channels, mode, patch_size=2, hidden_size=1024, cond_latent_channels=16,
                 cond_hidden=(32, 64, 128, 256)):
        super().__init__()
        assert mode in ("concat", "token")
        self.mode = mode
        self.stack, c_last = _cond_conv_stack(cond_channels, cond_hidden)
        if mode == "concat":
            self.head = nn.Conv2d(c_last, cond_latent_channels, 1)
        else:
            self.head = nn.Conv2d(c_last, hidden_size, patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.head.weight.view(self.head.weight.shape[0], -1))
        nn.init.zeros_(self.head.bias)

    def forward(self, sample, timestep, encoder_hidden_states=None, controlnet_cond=None, return_dict=True):
        with _autocast(sample.device):
            f = self.head(self.stack(controlnet_cond))
            cond = f.float() if self.mode == "concat" else f.flatten(2).transpose(1, 2).float()
        return _Out(down_block_res_samples=(cond,), mid_block_res_sample=None)


class DiTConcatBackbone(DiTBackbone):
    """z_t [4 ch] + encoded condition [cond_latent_channels] concatenated along channels."""

    def __init__(self, cond_latent_channels=16, **kw):
        super().__init__(in_channels=4 + cond_latent_channels, **kw)
        self.cond_latent_channels = cond_latent_channels

    def forward(self, sample, timestep, encoder_hidden_states=None, down_block_additional_residuals=None,
                mid_block_additional_residual=None, return_dict=True):
        cond = down_block_additional_residuals[0]
        assert cond.shape[-2:] == sample.shape[-2:], "condition must be at latent resolution"
        return super().forward(torch.cat([sample, cond], dim=1), timestep, encoder_hidden_states,
                               None, None, return_dict)


class DiTTokenBackbone(DiTBackbone):
    """Latent tokens [X] and condition tokens [C] concatenated along the sequence: joint attention
    over [X; C], same RoPE positions for aligned X_i / C_i, output head on X only."""

    def forward(self, sample, timestep, encoder_hidden_states=None, down_block_additional_residuals=None,
                mid_block_additional_residual=None, return_dict=True):
        B, _, H, W = sample.shape
        assert H == W and H % self.patch_size == 0
        grid = H // self.patch_size
        ctok = down_block_additional_residuals[0]
        n = grid * grid
        assert ctok.shape[1] == n, f"condition tokens {ctok.shape[1]} != latent tokens {n}"

        def rope(q):    # q: (B, heads, 2n, head_dim); condition tokens share the latent tokens' positions
            return torch.cat([self.rope(q[:, :, :n], grid), self.rope(q[:, :, n:], grid)], dim=2)

        with _autocast(sample.device):
            x, c, c0 = self.embed(sample, timestep)
            x = torch.cat([x, ctok.to(x.dtype)], dim=1)
            for blk in self.blocks:
                x = blk(x, c0, None, rope)
            out = self.unpatchify(self.final_layer(x[:, :n], c)).float()
        return _Out(sample=out)
