from __future__ import annotations

import torch

from grip.data import StreamSample

from .bypass_types import BypassProbeConfig, BypassProbeConfigError


def validate_config(config: BypassProbeConfig) -> None:
    if config.n_train_streams < 1 or config.n_test_streams < 1:
        raise BypassProbeConfigError("bypass probe needs at least one train and one test stream")
    if config.window < 1 or any(window < 1 for window in config.window_grid):
        raise BypassProbeConfigError("bypass probe window values must be positive")
    if config.ridge <= 0.0 or any(ridge <= 0.0 for ridge in config.ridge_grid):
        raise BypassProbeConfigError("bypass probe ridge values must be positive")
    if config.probe_epochs < 1:
        raise BypassProbeConfigError("bypass probe epochs must be positive")
    if config.answer_acc_threshold is not None and config.answer_acc_threshold <= 0.0:
        raise BypassProbeConfigError("answer accuracy threshold must be positive when provided")
    if config.answer_acc_chance_multiplier <= 0.0:
        raise BypassProbeConfigError("answer accuracy chance multiplier must be positive")
    if config.positive_control_r2_threshold <= 0.0:
        raise BypassProbeConfigError("positive control R2 threshold must be positive")
    if config.answer_convergence_min_delta < 0.0:
        raise BypassProbeConfigError("answer convergence min delta must be non-negative")
    train_stop = config.train_seed_base + config.n_train_streams
    test_stop = config.test_seed_base + config.n_test_streams
    if max(config.train_seed_base, config.test_seed_base) < min(train_stop, test_stop):
        raise BypassProbeConfigError("train and test seed ranges must be disjoint")


def candidate_windows(config: BypassProbeConfig) -> tuple[int, ...]:
    return tuple(sorted({config.window, *config.window_grid}))


def candidate_ridges(config: BypassProbeConfig) -> tuple[float, ...]:
    return tuple(sorted({config.ridge, *config.ridge_grid}))


def window_features(
    tokens: torch.Tensor | list[int],
    vocab_size: int,
    window: int,
    t: int,
) -> torch.Tensor:
    token_tensor = torch.as_tensor(tokens, dtype=torch.long)
    feature = torch.zeros(window, vocab_size, dtype=torch.float32)
    start = max(0, t - window + 1)
    selected = token_tensor[start : t + 1]
    offset = window - int(selected.shape[0])
    for idx, tok in enumerate(selected.tolist()):
        feature[offset + idx, int(tok)] = 1.0
    return feature.reshape(-1)


def answer_leak_features(sample: StreamSample, vocab_size: int, window: int) -> torch.Tensor:
    natural_len = int(sample.metadata["natural_len"])
    feature_len = min(window, natural_len)
    feature = torch.zeros(window, vocab_size, dtype=torch.float32)
    tokens = torch.as_tensor(sample.tokens[:feature_len], dtype=torch.long)
    for idx, tok in enumerate(tokens.tolist()):
        feature[idx, int(tok)] = 1.0
    return feature.reshape(-1)
