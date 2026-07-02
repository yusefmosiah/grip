"""Dense causal transformer — the quality upper reference.

Tiny, fully inspectable, MPS-friendly. Uses scaled_dot_product_attention (the
MPS SDPA path) — no custom kernels, no Triton, no FlexAttention here. This is
the frozen-backbone substrate for the derivative probe (SPEC-000).

Exposes hidden states so the probe can read them. Two heads: LM logits and an
auxiliary posterior head (the supervision channel; safe because the level
target is the control, the derivative is the experiment).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .outputs import DenseModelOutput
from .transformer_blocks import CausalTransformerBlock, CausalTransformerBlockConfig


class DenseTransformer(nn.Module):
    """Vanilla decoder-only transformer.

    Args:
        vocab_size, d_model, n_heads, n_layers, max_seq_len, n_hypotheses.
        n_hypotheses: size of the auxiliary posterior head output.

    forward(tokens) -> DenseModelOutput:
        lm_logits:  [B, T, vocab_size]
        posterior:  [B, T, n_hypotheses]   (softmax of the aux head)
        hidden:     [B, T, d_model]        (exposed for the probe — frozen backbone)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 512,
        n_hypotheses: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    CausalTransformerBlockConfig(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=4 * d_model,
                        dropout=dropout,
                    )
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie input/output embeddings (standard; saves params and regularizes)
        self.lm_head.weight = self.tok.weight
        self.aux_posterior = nn.Linear(d_model, n_hypotheses)

    def forward(self, tokens: torch.Tensor, real_mask: torch.Tensor | None = None) -> DenseModelOutput:
        B, T = tokens.shape
        if T > self.max_seq_len:
            msg = "sequence length exceeds max_seq_len"
            raise ValueError(msg)
        pos = torch.arange(T, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, real_mask=real_mask)
        x = self.norm_f(x)
        return DenseModelOutput(
            lm_logits=self.lm_head(x),  # [B,T,V]
            posterior=F.softmax(self.aux_posterior(x), dim=-1),  # [B,T,K]
            hidden=x,  # [B,T,d_model]
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
