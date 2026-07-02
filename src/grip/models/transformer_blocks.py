from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class CausalTransformerBlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    window: int | None = None
    dropout: float = 0.0


class CausalTransformerBlock(nn.Module):
    def __init__(self, config: CausalTransformerBlockConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            msg = "d_model must be divisible by n_heads"
            raise ValueError(msg)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.window = config.window
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, real_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_mask = attention_mask(
            seq_len=seq_len,
            device=x.device,
            window=self.window,
            real_mask=real_mask,
        )
        if attn_mask is None:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
        else:
            attn = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = x + self.drop(self.out(attn))
        return x + self.drop(self.ff(self.norm2(x)))


def attention_mask(
    *,
    seq_len: int,
    device: torch.device,
    window: int | None,
    real_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if window is None and real_mask is None:
        return None
    positions = torch.arange(seq_len, device=device)
    query_pos = positions[:, None]
    key_pos = positions[None, :]
    mask = key_pos <= query_pos
    if window is not None:
        mask = mask & (key_pos >= query_pos - window + 1)
    if real_mask is not None:
        mask = mask.reshape(1, 1, seq_len, seq_len) & real_mask.reshape(real_mask.shape[0], 1, 1, seq_len)
    return mask
