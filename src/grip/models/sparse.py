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
from typing import assert_never

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_components import (
    LocalBlockConfig,
    LocalCausalBlock,
    SparseAttentionMode,
    SparseConfigError,
    SparseMetadata,
    current_blocks,
    future_block_mask,
    parse_attention_mode,
)


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
        attention_mode: str | SparseAttentionMode = SparseAttentionMode.CONTENT_SPARSE,
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
        self.attention_mode = parse_attention_mode(attention_mode)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                LocalCausalBlock(
                    LocalBlockConfig(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=4 * d_model,
                        window=window,
                    )
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.content_projection = nn.Linear(d_model, d_model)
        self.content_norm = nn.LayerNorm(d_model)
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

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor | SparseMetadata | None]:
        """-> dict with 'lm_logits','posterior','hidden', and 'selected_blocks'[B,T,top_k]."""
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise SparseConfigError("tokens", "sequence length exceeds max_seq_len")
        pos = torch.arange(seq_len, device=tokens.device)
        hidden = self.tok(tokens) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm_f(hidden)
        block_summaries = self._summarize_blocks(hidden)
        selection_scores, selected_blocks = self._select_blocks(hidden, block_summaries)
        hidden = self._apply_attention_mode(hidden, block_summaries, selected_blocks)
        return {
            "lm_logits": self.lm_head(hidden),
            "posterior": F.softmax(self.aux_posterior(hidden), dim=-1),
            "hidden": hidden,
            "selected_blocks": selected_blocks,
            "selection_scores": selection_scores,
            "metadata": {
                "attention_mode": self.attention_mode.value,
                "block_size": self.block_size,
                "read_budget": self.top_k_blocks,
                "window": self.window,
            },
            "grip_state": None,
            "grip_recon": None,
        }

    def _select_blocks(
        self,
        hidden: torch.Tensor,
        block_summaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_summaries is None:
            block_summaries = self._summarize_blocks(hidden)
        scores = self._block_importance(hidden, block_summaries)
        masked_scores = scores.masked_fill(future_block_mask(hidden, self.block_size), float("-inf"))
        top_k = min(self.top_k_blocks, masked_scores.shape[-1])
        top_scores, top_blocks = torch.topk(masked_scores, k=top_k, dim=-1)
        query_blocks = current_blocks(hidden, self.block_size)
        top_blocks = torch.where(top_scores.isfinite(), top_blocks, query_blocks[..., None])
        if top_k == self.top_k_blocks:
            return masked_scores, top_blocks
        padding = query_blocks[..., None].expand(
            -1,
            -1,
            self.top_k_blocks - top_k,
        )
        return masked_scores, torch.cat([top_blocks, padding], dim=-1)

    def _apply_attention_mode(
        self,
        hidden: torch.Tensor,
        block_summaries: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        match self.attention_mode:
            case SparseAttentionMode.LOCAL:
                return hidden
            case SparseAttentionMode.CONTENT_SPARSE:
                context = self._selected_context(block_summaries, selected_blocks)
                return self.content_norm(hidden + self.content_projection(context))
            case unreachable:
                assert_never(unreachable)

    def _selected_context(
        self,
        block_summaries: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        channels = block_summaries.shape[-1]
        gather_index = selected_blocks[..., None].expand(-1, -1, -1, channels)
        selected = torch.gather(block_summaries, dim=2, index=gather_index)
        return selected.mean(dim=2)

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
