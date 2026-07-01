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
import torch
import torch.nn as nn


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
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks  # the read budget — log this in every run
        raise NotImplementedError(
            "CODEX: local-window attention over recent `window` tokens, plus "
            "top-K block selection over compressed block summaries. Start from "
            "lucidrains/native-sparse-attention-pytorch structure but keep a "
            "pure-PyTorch path. selection scoring MUST be a swappable method "
            "_block_importance() so grip variants can share the state producer "
            "and vary only lambda. Add tests in tests/test_sparse.py."
        )

    def _block_importance(self, query, block_summaries):
        """Score each block for selection. OVERRIDE POINT for grip variants.
        query: [B, T, d]; block_summaries: [B, num_blocks, d] -> [B, T, num_blocks]."""
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor):
        """-> dict with 'lm_logits','posterior','hidden', and 'selected_blocks'[B,T,top_k]."""
        raise NotImplementedError
