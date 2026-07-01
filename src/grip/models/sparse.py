"""Sparse attention models.

The content-sparse baseline: local sliding window + top-K content-block
selection, where block importance is scored by query-attending-to-compressed-
block-summaries (the NSA/DSA selection surface). This is the baseline grip
augments.

IMPLEMENTATION ROUTE:
  Local training must use pure PyTorch gather/masks or SDPA. Do NOT use Triton.
  Do NOT assume FlexAttention trains on MPS: the research notes found the
  current PyTorch FlexAttention path raises for MPS backward when inputs require
  gradients. Treat FlexAttention training as CUDA/cloud-only until upstream MPS
  backward support changes.

The grip-augmented variants (grip_read, grip_select) share an explicit Grip
state producer/update path. Their causal difference is ONLY whether that state
enters selection scoring. See GLOSSARY.md: the intervention surface is
  importance_b = f(q, c_b) + lambda * h(query_grip, grip_b)
with lambda=0 (variant A) vs lambda>0 (variant B).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class SparseConfigError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


class _LocalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, window: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise SparseConfigError("d_model", "must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch_size, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_mask = _local_causal_mask(seq_len, self.window, x.device)
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


def _local_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    query_pos = positions[:, None]
    key_pos = positions[None, :]
    return (key_pos <= query_pos) & (key_pos >= query_pos - window + 1)


class ContentSparseTransformer(nn.Module):
    """Local window + top-K content-block selection.

    Args: as DenseTransformer, plus:
        block_size: tokens per block.
        top_k_blocks: number of blocks selected per query (the READ BUDGET).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 512,
        n_hypotheses: int = 4,
        block_size: int = 32,
        top_k_blocks: int = 4,
        window: int = 32,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise SparseConfigError("vocab_size", "must be positive")
        if d_model <= 0:
            raise SparseConfigError("d_model", "must be positive")
        if n_heads <= 0:
            raise SparseConfigError("n_heads", "must be positive")
        if d_model % n_heads != 0:
            raise SparseConfigError("d_model", "must be divisible by n_heads")
        if n_layers <= 0:
            raise SparseConfigError("n_layers", "must be positive")
        if max_seq_len <= 0:
            raise SparseConfigError("max_seq_len", "must be positive")
        if n_hypotheses <= 0:
            raise SparseConfigError("n_hypotheses", "must be positive")
        if block_size <= 0:
            raise SparseConfigError("block_size", "must be positive")
        if top_k_blocks <= 0:
            raise SparseConfigError("top_k_blocks", "must be positive")
        if window <= 0:
            raise SparseConfigError("window", "must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.window = window
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_LocalBlock(d_model, n_heads, 4 * d_model, window) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.aux_posterior = nn.Linear(d_model, n_hypotheses)

    def _block_importance(
        self,
        query: torch.Tensor,
        block_summaries: torch.Tensor,
    ) -> torch.Tensor:
        """Score each block for selection. OVERRIDE POINT for grip variants."""
        scale = query.shape[-1] ** -0.5
        return torch.einsum("btd,btnd->btn", query, block_summaries) * scale

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """-> dict with 'lm_logits','posterior','hidden', and 'selected_blocks'[B,T,top_k]."""
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise SparseConfigError("tokens", "sequence length exceeds max_seq_len")
        pos = torch.arange(seq_len, device=tokens.device)
        hidden = self.tok(tokens) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm_f(hidden)
        selection_scores, selected_blocks = self._select_blocks(hidden)
        return {
            "lm_logits": self.lm_head(hidden),
            "posterior": F.softmax(self.aux_posterior(hidden), dim=-1),
            "hidden": hidden,
            "selected_blocks": selected_blocks,
            "selection_scores": selection_scores,
            "grip_state": None,
            "grip_recon": None,
        }

    def _select_blocks(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_summaries = self._summarize_blocks(hidden)
        scores = self._block_importance(hidden, block_summaries)
        masked_scores = scores.masked_fill(self._future_block_mask(hidden), float("-inf"))
        top_k = min(self.top_k_blocks, masked_scores.shape[-1])
        top_scores, top_blocks = torch.topk(masked_scores, k=top_k, dim=-1)
        current_blocks = self._current_blocks(hidden)
        top_blocks = torch.where(top_scores.isfinite(), top_blocks, current_blocks[..., None])
        if top_k == self.top_k_blocks:
            return masked_scores, top_blocks
        padding = current_blocks[..., None].expand(
            -1,
            -1,
            self.top_k_blocks - top_k,
        )
        return masked_scores, torch.cat([top_blocks, padding], dim=-1)

    def _summarize_blocks(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        positions = torch.arange(seq_len, device=hidden.device)
        block_ids = positions // self.block_size
        summary_block_ids = torch.arange(num_blocks, device=hidden.device)
        key_positions = positions.reshape(1, 1, seq_len)
        query_positions = positions.reshape(seq_len, 1, 1)
        key_blocks = block_ids.reshape(1, 1, seq_len)
        summary_blocks = summary_block_ids.reshape(1, num_blocks, 1)
        include = (key_positions <= query_positions) & (key_blocks == summary_blocks)
        weights = include.to(hidden.dtype)
        totals = torch.einsum("bkd,qnk->bqnd", hidden, weights)
        counts = weights.sum(dim=-1).clamp_min(1).reshape(1, seq_len, num_blocks, 1)
        return totals / counts

    def _current_blocks(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        positions = torch.arange(seq_len, device=hidden.device)
        return (positions // self.block_size).expand(hidden.shape[0], seq_len)

    def _future_block_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        current_blocks = self._current_blocks(hidden)
        num_blocks = (hidden.shape[1] + self.block_size - 1) // self.block_size
        block_ids = torch.arange(num_blocks, device=hidden.device)
        return block_ids.reshape(1, 1, num_blocks) > current_blocks[..., None]
