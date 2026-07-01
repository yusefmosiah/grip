from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class SparseConfigError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


class SparseAttentionMode(StrEnum):
    LOCAL = "local"
    CONTENT_SPARSE = "content_sparse"


class SparseMetadata(TypedDict):
    attention_mode: str
    block_size: int
    read_budget: int
    window: int


@dataclass(frozen=True, slots=True)
class LocalBlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    window: int


class LocalCausalBlock(nn.Module):
    def __init__(self, config: LocalBlockConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise SparseConfigError("d_model", "must be divisible by n_heads")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_mask = local_causal_mask(seq_len, self.window, x.device)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = x + self.out(attn)
        return x + self.ff(self.norm2(x))


def local_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    query_pos = positions[:, None]
    key_pos = positions[None, :]
    return (key_pos <= query_pos) & (key_pos >= query_pos - window + 1)


def current_blocks(hidden: torch.Tensor, block_size: int) -> torch.Tensor:
    seq_len = hidden.shape[1]
    positions = torch.arange(seq_len, device=hidden.device)
    return (positions // block_size).expand(hidden.shape[0], seq_len)


def future_block_mask(hidden: torch.Tensor, block_size: int) -> torch.Tensor:
    query_blocks = current_blocks(hidden, block_size)
    num_blocks = (hidden.shape[1] + block_size - 1) // block_size
    block_ids = torch.arange(num_blocks, device=hidden.device)
    return block_ids.reshape(1, 1, num_blocks) > query_blocks[..., None]


def parse_attention_mode(raw: str | SparseAttentionMode) -> SparseAttentionMode:
    try:
        return SparseAttentionMode(raw)
    except ValueError as exc:
        raise SparseConfigError("attention_mode", "must be local or content_sparse") from exc
