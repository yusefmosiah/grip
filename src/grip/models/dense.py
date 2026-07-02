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


class _Block(nn.Module):
    """Pre-norm transformer block with causal SDPA attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, real_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)  # [B, n_heads, T, 3*d_head]
        q, k, v = qkv.chunk(3, dim=-1)
        if real_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            attn_mask = _causal_key_mask(real_mask, T)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.out(attn))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


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
            [_Block(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
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


def _causal_key_mask(real_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len, device=real_mask.device)
    causal = positions[None, :] <= positions[:, None]
    return causal.reshape(1, 1, seq_len, seq_len) & real_mask.reshape(real_mask.shape[0], 1, 1, seq_len)
