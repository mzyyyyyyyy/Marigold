"""
LightningDiT (VOSR variant) -- compact re-implementation for loading the public
VOSR checkpoints (github.com/cswry/VOSR, Apache-2.0; weights: huggingface.co/CSWRY/VOSR).

Architecture/code is adapted from VOSR's models/lightningdit.py, which in turn builds on
LightningDiT (hustvl) / DiT (facebookresearch) / SiT. Parameter names match the VOSR
checkpoints exactly. Differences from the original file, none of which change the math:
  * no timm / einops / fairscale dependency (PatchEmbed, Mlp, rotary helpers are inlined)
  * no torch.compile decorators
  * RoPE for any square token grid is built on the fly (VOSR's forward_flexible: positions are
    rescaled to the pretraining grid), optional gradient checkpointing per block

Block: adaLN (scale/shift table + timestep embedding), QK-normed RoPE self-attention, optional
cross-attention to an external token set `z`, SwiGLU MLP, RMSNorm.
Input x is a channel-concatenation of [conditioning latent(s), noisy latent]; output is the
velocity (out_channels). `z` is a list whose first element is a (B, N, z_dims) token tensor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """2D RoPE (EVA-02 style). `forward(t, grid)` rotates (B, heads, N, head_dim) for an N = grid*grid
    token grid; positions are rescaled to the pretraining grid `pt_seq_len`."""

    def __init__(self, dim: int, pt_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim, self.pt_seq_len, self.theta = dim, pt_seq_len, theta
        cos, sin = self._tables(pt_seq_len, torch.device("cpu"))
        # kept as buffers so VOSR checkpoints load without "unexpected key" noise
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    def _tables(self, grid: int, device):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device)[: self.dim // 2].float() / self.dim))
        pos = torch.arange(grid, device=device).float() / grid * self.pt_seq_len
        f = (pos[:, None] * freqs[None]).repeat_interleave(2, dim=-1)           # (grid, dim)
        f = torch.cat([f[:, None, :].expand(grid, grid, -1), f[None, :, :].expand(grid, grid, -1)], dim=-1)
        f = f.reshape(grid * grid, -1)                                           # (grid^2, 2*dim)
        return f.cos(), f.sin()

    def forward(self, t, grid: int):
        cos, sin = self._tables(grid, t.device)
        return t * cos.to(t.dtype) + _rotate_half(t) * sin.to(t.dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, cond):
        B, N, C = x.shape
        Nc = cond.shape[1]
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(cond).view(B, Nc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(B, Nc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, N, C))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, z_dims=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * int(hidden_size * mlp_ratio)))
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.z_dims = z_dims
        if z_dims is not None:
            self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=True)

    def forward(self, x, c, z=None, rope=None):
        B = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa, rope=rope)
        if self.z_dims is not None and z is not None:
            x = x + self.cross_attn(x, z)
        return x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class LightningDiT(nn.Module):
    def __init__(self, input_size=64, patch_size=2, in_channels=8, out_channels=4, hidden_size=1024,
                 depth=28, num_heads=16, mlp_ratio=4.0, z_dims=None, encdim_ratio=3, use_checkpoint=False):
        super().__init__()
        self.in_channels, self.out_channels, self.patch_size = in_channels, out_channels, patch_size
        self.num_heads, self.depth, self.hidden_size = num_heads, depth, hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=hidden_size // num_heads // 2,
                                                   pt_seq_len=input_size // patch_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.blocks = nn.ModuleList([LightningDiTBlock(hidden_size, num_heads, mlp_ratio, z_dims)
                                     for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.z_dims = z_dims
        if z_dims is not None:
            self.layer_norm = nn.LayerNorm(z_dims)
            self.mlp_ca = Mlp(z_dims, hidden_size * encdim_ratio, hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        for p in (self.final_layer.adaLN_modulation[-1].weight, self.final_layer.adaLN_modulation[-1].bias,
                  self.final_layer.linear.weight, self.final_layer.linear.bias):
            nn.init.constant_(p, 0)

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, t, z=None):
        """x: (N, in_channels, H, W) square latent; t: (N,) in [0,1]; z: list with one (N, Nz, z_dims) tensor."""
        N, _, H, W = x.shape
        assert H == W, "square inputs only"
        grid = H // self.patch_size
        x = self.x_embedder(x)
        c = self.t_embedder(t)
        c0 = self.t_block(c)
        if self.z_dims is not None and z is not None:
            z = self.mlp_ca(self.layer_norm(z[0]))
        else:
            z = None
        rope = lambda q: self.feat_rope(q, grid)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, c0, z, rope, use_reentrant=False)
            else:
                x = blk(x, c0, z, rope)
        return self.unpatchify(self.final_layer(x, c))
