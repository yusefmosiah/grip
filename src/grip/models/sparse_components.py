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
    GRIP_READ = "grip_read"
    GRIP_SELECT = "grip_select"


class SparseMetadata(TypedDict):
    attention_mode: str
    block_size: int
    read_budget: int
    window: int


@dataclass(frozen=True, slots=True)
class CausalBlockSummaries:
    full: torch.Tensor
    current_prefix: torch.Tensor
    token_blocks: torch.Tensor


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

    def forward(self, x: torch.Tensor, real_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_mask = local_causal_mask(seq_len, self.window, x.device)
        if real_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, seq_len, seq_len) & real_mask.reshape(batch_size, 1, 1, seq_len)
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


def causal_block_summaries(
    hidden: torch.Tensor,
    block_size: int,
    real_mask: torch.Tensor | None = None,
) -> CausalBlockSummaries:
    batch_size, seq_len, channels = hidden.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    positions = torch.arange(seq_len, device=hidden.device)
    block_ids = positions // block_size
    if real_mask is None:
        real_weights = hidden.new_ones(batch_size, seq_len)
    else:
        real_weights = real_mask.to(device=hidden.device, dtype=hidden.dtype)
    masked_hidden = hidden * real_weights.unsqueeze(-1)
    block_totals = hidden.new_zeros(batch_size, num_blocks, channels)
    block_totals.index_add_(1, block_ids, masked_hidden)
    block_counts = hidden.new_zeros(batch_size, num_blocks)
    block_counts.index_add_(1, block_ids, real_weights)
    full = block_totals / block_counts.clamp_min(1).reshape(batch_size, num_blocks, 1)
    prefix_totals = masked_hidden.cumsum(dim=1)
    prefix_counts = real_weights.cumsum(dim=1)
    block_starts = block_ids * block_size
    prior_index = (block_starts - 1).clamp_min(0)
    prior_totals = prefix_totals[:, prior_index, :]
    prior_counts = prefix_counts[:, prior_index]
    has_prior = (block_starts > 0).reshape(1, seq_len, 1)
    zero_prior = torch.zeros_like(prior_totals)
    current_totals = prefix_totals - torch.where(has_prior, prior_totals, zero_prior)
    current_counts = prefix_counts - torch.where(
        has_prior.squeeze(-1),
        prior_counts,
        torch.zeros_like(prior_counts),
    )
    current_prefix = current_totals / current_counts.clamp_min(1).reshape(batch_size, seq_len, 1)
    return CausalBlockSummaries(full=full, current_prefix=current_prefix, token_blocks=block_ids)


def selected_block_context(
    block_summaries: CausalBlockSummaries,
    selection_scores: torch.Tensor,
    selected_blocks: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = selected_blocks.shape
    batch_index = torch.arange(batch_size, device=selected_blocks.device).reshape(batch_size, 1, 1)
    selected = block_summaries.full[batch_index, selected_blocks]
    current_blocks_for_token = block_summaries.token_blocks.reshape(1, seq_len, 1)
    current_prefix = block_summaries.current_prefix[:, :, None, :]
    selected = torch.where(
        (selected_blocks == current_blocks_for_token)[..., None],
        current_prefix,
        selected,
    )
    selected_scores = torch.gather(selection_scores, dim=2, index=selected_blocks)
    weights = F.softmax(selected_scores, dim=-1).unsqueeze(-1)
    return (selected * weights).sum(dim=2)


def causal_block_scores(
    query_projection: torch.Tensor,
    full_key_projection: torch.Tensor,
    current_key_projection: torch.Tensor,
    block_summaries: CausalBlockSummaries,
) -> torch.Tensor:
    scale = query_projection.shape[-1] ** -0.5
    scores = torch.einsum("btd,bnd->btn", query_projection, full_key_projection) * scale
    current_scores = (query_projection * current_key_projection).sum(dim=-1) * scale
    current_index = block_summaries.token_blocks.reshape(1, -1, 1).expand(query_projection.shape[0], -1, 1)
    return scores.scatter(dim=2, index=current_index, src=current_scores[..., None])


def parse_attention_mode(raw: str | SparseAttentionMode) -> SparseAttentionMode:
    try:
        return SparseAttentionMode(raw)
    except ValueError as exc:
        raise SparseConfigError("attention_mode", "must be local, content_sparse, grip_read, or grip_select") from exc
