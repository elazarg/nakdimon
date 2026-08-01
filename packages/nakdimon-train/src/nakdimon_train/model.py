"""Nakdimon v2 model: conv stem + pre-norm Transformer encoder with RoPE.

Same interface as v1 (char ids in; niqqud/dagesh/sin logits out), new body.
Export constraints (see export.py): no data-dependent control flow, RoPE tables
computed from the runtime sequence length, SDPA attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    # Vocabulary sizes: spec/hebrew.json inventories.v2 (letters 43, niqqud 15,
    # dagesh 3, sin 4). Kept explicit here so a checkpoint is self-describing.
    n_letters: int = 43
    n_niqqud: int = 15
    n_dagesh: int = 3
    n_sin: int = 4

    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 3
    conv_blocks: int = 2
    conv_kernel: int = 5
    dropout: float = 0.1
    max_context: int = 1024  # advisory; RoPE has no hard limit


def rope_tables(head_dim: int, length: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(length, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class ConvBlock(nn.Module):
    """Depthwise-separable conv block: local n-gram features chars want."""

    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        h = self.norm(x).transpose(1, 2)
        h = self.pointwise(F.gelu(self.depthwise(h))).transpose(1, 2)
        return x + self.dropout(h)


class SelfAttention(nn.Module):
    """Explicit matmul+softmax attention (no SDPA): trivially ONNX-exportable
    with any exporter, which matters more than fused kernels at this scale."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_bias: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(z: Tensor) -> Tensor:
            return z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) * self.head_dim**-0.5 + attn_bias
        probs = self.dropout(torch.softmax(scores, dim=-1))
        out = probs @ v
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.mlp_norm = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, attn_bias: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), attn_bias, cos, sin))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class NakdimonV2(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.embed = nn.Embedding(cfg.n_letters, cfg.d_model, padding_idx=0)
        self.convs = nn.ModuleList(ConvBlock(cfg.d_model, cfg.conv_kernel, cfg.dropout) for _ in range(cfg.conv_blocks))
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head_n = nn.Linear(cfg.d_model, cfg.n_niqqud)
        self.head_d = nn.Linear(cfg.d_model, cfg.n_dagesh)
        self.head_s = nn.Linear(cfg.d_model, cfg.n_sin)

    def features(self, ids: Tensor) -> Tensor:
        """Backbone shared by diacritization fine-tuning and masked-char
        pretraining (pretrain.py adds a letters head on top of this)."""
        # Accept int32 (the runtime/ONNX contract, shared with the v1 model);
        # the cast is exported into the graph so sessions take int32 directly.
        ids = ids.long()
        x = self.embed(ids)
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = rope_tables(head_dim, ids.shape[1], ids.device, x.dtype)
        # Additive attention bias: large-negative (not -inf, which turns padding
        # queries into NaN rows) at padded keys.
        pad = (ids == 0)[:, None, None, :]
        attn_bias = torch.zeros(pad.shape, dtype=x.dtype, device=x.device).masked_fill(pad, -1e9)
        for conv in self.convs:
            x = conv(x)
        for layer in self.layers:
            x = layer(x, attn_bias, cos, sin)
        return self.final_norm(x)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.features(ids)
        return self.head_n(x), self.head_d(x), self.head_s(x)


def masked_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """CE over positions where target != 0 (mask/pad class), mirroring v1's
    ignore_class=0 so RAFE stays a real, supervised label."""
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=0)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
