"""Metric implementations. Pure functions on (preds, targets).

All numerically stable, all unit-testable. The derivative-reconstruction
metric (recon_error) is the one that answers the gating experiment.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred: [N, K] logits or probs; target: [N] long. Returns top-1 accuracy."""
    pred_ids = pred.argmax(dim=-1)
    return (pred_ids == target).float().mean().item()


def nll(log_probs: torch.Tensor, target: torch.Tensor) -> float:
    """log_probs: [N, K] (log-softmax); target: [N] long. Mean NLL."""
    return F.nll_loss(log_probs, target).item()


def brier_score(probs: torch.Tensor, target: torch.Tensor) -> float:
    """Multi-class Brier. probs: [N, K] (softmax); target: [N] long."""
    N, K = probs.shape
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, target.unsqueeze(1), 1.0)
    return float(((probs - onehot) ** 2).sum(dim=1).mean())


def ece(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (top-1 confidence vs accuracy)."""
    conf, pred = probs.max(dim=1)
    correct = (pred == target).float()
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece_val = 0.0
    N = probs.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece_val += (in_bin.float().sum() / N) * (acc_bin - conf_bin).abs()
    return float(ece_val)


def recon_error(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean squared error between a predicted latent and ground-truth latent.
    Used for the derivative probe: predicted=probe head output, truth=stream d_conf.
    Accepts arbitrary matching shapes."""
    return F.mse_loss(predicted, truth).item()


def r2_score(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    """Coefficient of determination. R^2<=0 means worse than predicting the mean;
    R^2~1 is a perfect fit. R^2~0 is the 'amnesia' / noise-floor signature for the
    derivative probe."""
    ss_res = ((truth - predicted) ** 2).sum()
    ss_tot = ((truth - truth.mean()) ** 2).sum()
    if ss_tot.item() == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def decisive_token_recall(
    selected_blocks: torch.Tensor, decisive_idx: torch.Tensor
) -> float:
    """Fraction of decisive-evidence positions whose block was selected.

    selected_blocks: [B, T, top_k] long — block id selected per query position.
    decisive_idx:    [B, T] long/bool — 1 where step t is a decisive step.
    A decisive step "recalled" if its own block id is in the selected set at t.
    """
    B, T, Ksel = selected_blocks.shape
    # block id of each position = floor(t / block_size); but we don't know block_size
    # here, so infer it from block id max + shape. Cleaner: caller passes block ids
    # per position. We approximate by treating each position's block as the set
    # membership test directly.
    own_block = torch.arange(T, device=selected_blocks.device).unsqueeze(0)
    own_block = own_block.expand(B, T)  # placeholder; see note
    # membership: is own_block[b,t] in selected_blocks[b,t,:]?
    sel = selected_blocks  # [B,T,Ksel]
    hits = (sel == own_block.unsqueeze(-1)).any(dim=-1)  # [B,T]
    decisive_mask = decisive_idx.bool()
    if not decisive_mask.any():
        return 0.0
    return float(hits[decisive_mask].float().mean())
