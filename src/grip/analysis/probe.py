"""Linear probes for the gating experiment.

Frozen backbone, train only a linear (or small MLP) head from hidden states to
each target latent. Report R^2 / MSE per target. The split: probe trains on a
held-out split of streams — never the same streams used to train the backbone.

CRITICAL: the probe must NOT be able to trivially win by reading the answer.
The level targets (posterior, entropy) will probe well — that's the control.
The derivative targets are the experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ProbeResult:
    target_name: str
    mse: float
    r2: float
    n_train: int
    n_test: int


def linear_probe(
    hidden: torch.Tensor,        # [N, T, d_model] frozen hidden states
    target: torch.Tensor,        # [N, T] or [N, T, k] ground-truth latent
    target_name: str,
    train_mask: torch.Tensor,    # [N, T] bool, which (n,t) are train
    n_epochs: int = 200,
    lr: float = 1e-2,
    device: str = "mps",
) -> ProbeResult:
    """Fit a linear probe from hidden -> target on train_mask, eval on the rest.

    Returns ProbeResult. This is the entire gating experiment for one target.
    """
    raise NotImplementedError(
        "CODEX: linear nn.Linear(d_model, target_dim). Train with Adam on the "
        "train_mask positions, eval R^2 and MSE on held-out positions. Must be "
        "fast (this runs 4 targets x a few seeds = the gating decision). "
        "Add tests in tests/test_probe.py."
    )
