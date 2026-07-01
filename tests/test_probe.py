"""Probe sanity tests: it must recover a linear signal and reject noise."""
import torch
import pytest

from grip.analysis.probe import ProbeSplitError, _flat_probe, linear_probe


def test_probe_recovers_linear_signal():
    torch.manual_seed(0)
    P, d = 2000, 64
    H = torch.randn(P, d)
    w = torch.randn(d)
    target = H @ w + 0.01 * torch.randn(P)  # near-perfect linear signal
    is_train = torch.arange(P) < 1600
    mse, r2 = _flat_probe(H, target, is_train)
    assert r2 > 0.9, f"probe should recover linear signal, R^2={r2:.3f}"


def test_probe_rejects_noise():
    torch.manual_seed(0)
    P, d = 2000, 64
    H = torch.randn(P, d)
    target = torch.randn(P)  # pure noise, independent of H
    is_train = torch.arange(P) < 1600
    mse, r2 = _flat_probe(H, target, is_train)
    assert r2 < 0.05, f"probe should not fit noise, R^2={r2:.3f}"


def test_probe_train_test_disjoint():
    # ensure eval is on the held-out positions, not train
    torch.manual_seed(0)
    P, d = 1000, 32
    H = torch.randn(P, d)
    target = H[:, 0]  # perfectly linear in one dim
    is_train = torch.zeros(P, dtype=torch.bool)
    is_train[:800] = True
    _, r2 = _flat_probe(H, target, is_train)
    assert r2 > 0.85


def test_public_linear_probe_evaluates_on_held_out_complement():
    torch.manual_seed(0)
    n_streams, n_steps, d_model = 20, 16, 32
    hidden = torch.randn(n_streams, n_steps, d_model)
    target = hidden[..., 0] + 0.01 * torch.randn(n_streams, n_steps)
    train_mask = torch.zeros(n_streams, n_steps, dtype=torch.bool)
    train_mask[:14] = True

    result = linear_probe(hidden, target, "linear_signal", train_mask)

    assert result.n_train == 14 * n_steps
    assert result.n_test == 6 * n_steps
    assert result.r2 > 0.85


def test_public_linear_probe_rejects_empty_test_split():
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 8)
    target = hidden[..., 0]
    train_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ProbeSplitError):
        linear_probe(hidden, target, "all_train", train_mask)
