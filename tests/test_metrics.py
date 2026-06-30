"""Unit tests for metrics — numerical correctness on known inputs."""
import torch
from grip.eval import metrics as M


def test_accuracy_perfect():
    p = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    t = torch.tensor([1, 0])
    assert M.accuracy(p, t) == 1.0


def test_accuracy_half():
    p = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    t = torch.tensor([0, 0])  # one right, one wrong
    assert abs(M.accuracy(p, t) - 0.5) < 1e-6


def test_brier_zero_when_perfect_confident():
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = torch.tensor([0, 1])
    assert M.brier_score(p, t) < 1e-6


def test_brier_high_when_wrong_confident():
    p = torch.tensor([[1.0, 0.0]])
    t = torch.tensor([1])
    assert M.brier_score(p, t) > 1.5  # (0-1)^2 + (1-0)^2 = 2


def test_r2_perfect():
    y = torch.tensor([1.0, 2.0, 3.0])
    assert abs(M.r2_score(y, y) - 1.0) < 1e-5


def test_r2_zero_predicts_mean():
    # predicting the mean -> R^2 = 0
    y = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.full_like(y, y.mean())
    assert abs(M.r2_score(pred, y)) < 1e-5


def test_r2_negative_when_worse_than_mean():
    y = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([10.0, -5.0, 20.0])
    assert M.r2_score(pred, y) < 0


def test_recon_error_zero():
    a = torch.randn(4, 8)
    assert M.recon_error(a, a) < 1e-6


def test_ece_perfect_calibration():
    # perfectly calibrated: confidence == accuracy in every bin
    p = torch.tensor([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5)
    t = torch.tensor([0] * 4 + [1] * 1 + [0] * 1 + [1] * 4)  # acc in 0.9 bin = 0.9
    ece = M.ece(p, t, n_bins=10)
    assert ece < 0.1
