"""Sparse attention models with pure-PyTorch local/content and Grip A/B paths.

Grip A reads explicit Grip state without using it for selection; Grip B adds
Grip-state scores to the same selector. FlexAttention/Triton remain out of the
local MPS training path.
"""
from __future__ import annotations
from typing import assert_never

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_components import (
    CausalBlockSummaries,
    LocalBlockConfig,
    LocalCausalBlock,
    SparseAttentionMode,
    SparseConfigError,
    SparseMetadata,
    causal_block_scores,
    causal_block_summaries,
    current_blocks,
    future_block_mask,
    parse_attention_mode,
    selected_block_context,
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
        self.selector_query = nn.Linear(d_model, d_model, bias=False)
        self.selector_key = nn.Linear(d_model, d_model, bias=False)
        self.grip_state_projection = nn.Linear(d_model, d_model)
        self.grip_query = nn.Linear(d_model, d_model, bias=False)
        self.grip_key = nn.Linear(d_model, d_model, bias=False)
        self.grip_recon = nn.Linear(d_model, 2)
        self.grip_recon_projection = nn.Linear(2, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.aux_posterior = nn.Linear(d_model, n_hypotheses)
        self._initialize_selector()

    def _initialize_selector(self) -> None:
        nn.init.eye_(self.selector_query.weight)
        nn.init.eye_(self.selector_key.weight)
        nn.init.eye_(self.grip_query.weight)
        nn.init.eye_(self.grip_key.weight)

    def _block_importance(
        self,
        query: torch.Tensor,
        block_summaries: CausalBlockSummaries,
    ) -> torch.Tensor:
        """Score each block for selection. OVERRIDE POINT for grip variants."""
        return causal_block_scores(
            self.selector_query(query),
            self.selector_key(block_summaries.full),
            self.selector_key(block_summaries.current_prefix),
            block_summaries,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        real_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | SparseMetadata | None]:
        """-> dict with 'lm_logits','posterior','hidden', and 'selected_blocks'[B,T,top_k]."""
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise SparseConfigError("tokens", "sequence length exceeds max_seq_len")
        if real_mask is not None and real_mask.shape != tokens.shape:
            raise SparseConfigError("real_mask", "must match tokens shape")
        pos = torch.arange(seq_len, device=tokens.device)
        hidden = self.tok(tokens) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden, real_mask=real_mask)
        hidden = self.norm_f(hidden)
        exposes_grip = self._exposes_grip()
        grip_state = self._grip_state(hidden) if exposes_grip else None
        block_summaries = self._summarize_blocks(hidden, real_mask=real_mask)
        grip_summaries = (
            self._summarize_blocks(self._grip_read_features(grip_state), real_mask=real_mask)
            if grip_state is not None
            else None
        )
        selection_scores, selected_blocks = self._select_blocks(
            hidden,
            block_summaries,
            grip_state,
            grip_summaries,
        )
        hidden = self._apply_attention_mode(
            hidden,
            block_summaries,
            grip_summaries,
            selection_scores,
            selected_blocks,
        )
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
            "grip_state": grip_state if exposes_grip else None,
            "grip_recon": self.grip_recon(grip_state) if exposes_grip else None,
        }

    def _select_blocks(
        self,
        hidden: torch.Tensor,
        block_summaries: CausalBlockSummaries | None = None,
        grip_state: torch.Tensor | None = None,
        grip_summaries: CausalBlockSummaries | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_summaries is None:
            block_summaries = self._summarize_blocks(hidden)
        scores = self._selection_importance(hidden, block_summaries, grip_state, grip_summaries)
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
        block_summaries: CausalBlockSummaries,
        grip_summaries: CausalBlockSummaries | None,
        selection_scores: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        match self.attention_mode:
            case SparseAttentionMode.LOCAL:
                return hidden
            case SparseAttentionMode.CONTENT_SPARSE:
                context = self._selected_context(block_summaries, selection_scores, selected_blocks)
                return self.content_norm(hidden + self.content_projection(context))
            case SparseAttentionMode.GRIP_READ | SparseAttentionMode.GRIP_SELECT:
                if grip_summaries is None:
                    raise SparseConfigError("grip_summaries", "required for grip variants")
                context = self._selected_context(grip_summaries, selection_scores, selected_blocks)
                return self.content_norm(hidden + self.content_projection(context))
            case unreachable:
                assert_never(unreachable)

    def _selection_importance(
        self,
        hidden: torch.Tensor,
        block_summaries: CausalBlockSummaries,
        grip_state: torch.Tensor | None,
        grip_summaries: CausalBlockSummaries | None,
    ) -> torch.Tensor:
        scores = self._block_importance(hidden, block_summaries)
        if self.attention_mode != SparseAttentionMode.GRIP_SELECT:
            return scores
        if grip_state is None:
            grip_state = self._grip_state(hidden)
        if grip_summaries is None:
            grip_summaries = self._summarize_blocks(self._grip_read_features(grip_state))
        return scores + self._grip_importance(grip_state, grip_summaries)

    def _exposes_grip(self) -> bool:
        return self.attention_mode in {SparseAttentionMode.GRIP_READ, SparseAttentionMode.GRIP_SELECT}

    def _grip_state(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.grip_state_projection(hidden))

    def _grip_read_features(self, grip_state: torch.Tensor) -> torch.Tensor:
        return grip_state + self.grip_recon_projection(self.grip_recon(grip_state))

    def _grip_importance(
        self,
        query_grip: torch.Tensor,
        grip_summaries: CausalBlockSummaries,
    ) -> torch.Tensor:
        return causal_block_scores(
            self.grip_query(query_grip),
            self.grip_key(grip_summaries.full),
            self.grip_key(grip_summaries.current_prefix),
            grip_summaries,
        )

    def _selected_context(
        self,
        block_summaries: CausalBlockSummaries,
        selection_scores: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        return selected_block_context(block_summaries, selection_scores, selected_blocks)

    def _summarize_blocks(
        self,
        hidden: torch.Tensor,
        real_mask: torch.Tensor | None = None,
    ) -> CausalBlockSummaries:
        return causal_block_summaries(hidden, self.block_size, real_mask=real_mask)
