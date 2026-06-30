"""Metric implementations. Pure functions on (preds, targets).

All numerically stable. All tested in tests/test_metrics.py before use.
The derivative-reconstruction metric (recon_error on d_conf) is the one that
answers the gating experiment.
"""
from __future__ import annotations
import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred: [N, K] logits/probs; target: [N] int. Returns top-1 accuracy."""
    raise NotImplementedError("CODEX")


def nll(log_probs: torch.Tensor, target: torch.Tensor) -> float:
    """log_probs: [N, K] (log-softmax); target: [N] int."""
    raise NotImplementedError("CODEX")


def brier_score(probs: torch.Tensor, target: torch.Tensor) -> float:
    """probs: [N, K] (softmax); target: [N] int. Multi-class Brier."""
    raise NotImplementedError("CODEX")


def ece(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    raise NotImplementedError("CODEX")


def recon_error(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean squared error between a predicted latent and the ground-truth latent.
    Used for the derivative probe: predicted=d_conf head output, truth=stream d_conf."""
    raise NotImplementedError("CODEX")


def decisive_token_recall(selected_blocks: torch.Tensor, decisive_idx: torch.Tensor) -> float:
    """Fraction of decisive-evidence blocks that were selected by sparse attention.
    selected_blocks: [B, T, top_k] block indices; decisive_idx: [B, T] per-step
    decisive block id. The attention-quality metric."""
    raise NotImplementedError("CODEX")
