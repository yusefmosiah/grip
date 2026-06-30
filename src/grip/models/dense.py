"""Dense causal transformer — the quality upper reference.

Tiny, fully inspectable. MPS-friendly: prefer scaled_dot_product_attention
(MPS SDPA path) over any custom kernel here. No Triton, no FlexAttention yet.

The dense baseline exists so we can establish the selection-error regime:
grip can only help where content-sparse measurably underperforms dense.
Finding that regime is milestone M-regime (see DAG).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class DenseTransformer(nn.Module):
    """Vanilla decoder-only transformer.

    Args:
        vocab_size, d_model, n_heads, n_layers, max_seq_len, n_hypotheses.
        n_hypotheses: size of the auxiliary head output (for posterior pred).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 512,
        n_hypotheses: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        raise NotImplementedError(
            "CODEX: standard decoder-only transformer. Token + pos embed, "
            "n_layers of (causal SDPA attention + MLP), LayerNorm. Two heads: "
            "lm_logits (vocab) and aux_posterior (n_hypotheses, softmax). "
            "Keep it minimal and MPS-safe. Add tests in tests/test_dense.py."
        )

    def forward(self, tokens: torch.Tensor):
        """tokens: long[B,T] -> dict with 'lm_logits'[B,T,V], 'posterior'[B,T,K],
        and 'hidden'[B,T,d_model] (hidden states exposed for the probe)."""
        raise NotImplementedError
