from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from grip.data import StreamSample


class StreamLike(Protocol):
    T: int
    K: int
    vocab_size: int

    def generate(self, seed: int | None = None) -> StreamSample: ...


@dataclass(frozen=True, slots=True)
class BypassProbeConfig:
    n_train_streams: int = 200
    n_test_streams: int = 80
    train_seed_base: int = 30_000_000
    test_seed_base: int = 40_000_000
    window: int = 8
    probe_epochs: int = 300
    d_conf_r2_threshold: float = 0.05
    answer_acc_threshold: float | None = None
    answer_acc_chance_multiplier: float = 1.6
    ridge: float = 1e-3
    window_grid: tuple[int, ...] = (4, 8, 16)
    ridge_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    positive_control_r2_threshold: float = 0.95
    answer_convergence_min_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class BypassDataset:
    token_features: torch.Tensor
    d_conf: torch.Tensor
    train_position_mask: torch.Tensor
    answer_features: torch.Tensor
    answers: torch.Tensor
    train_stream_mask: torch.Tensor
    n_train_streams: int
    n_test_streams: int
    train_seed_base: int
    test_seed_base: int
    num_classes: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class BypassProbeResult:
    d_conf_mse: float
    d_conf_r2: float
    answer_accuracy: float
    n_train_positions: int
    n_test_positions: int
    n_train_streams: int
    n_test_streams: int
    window: int
    ridge: float
    answer_window: int
    window_grid: tuple[int, ...]
    ridge_grid: tuple[float, ...]
    d_conf_r2_threshold: float
    answer_acc_threshold: float
    positive_control_r2: float
    positive_control_r2_threshold: float
    positive_control_passed: bool
    answer_train_loss_initial: float
    answer_train_loss_final: float
    answer_converged: bool
    train_seed_base: int
    test_seed_base: int
    d_conf_passed: bool
    answer_passed: bool
    passed: bool


class BypassProbeConfigError(ValueError):
    pass
