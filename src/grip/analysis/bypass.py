from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from grip.data import StreamSample
from grip.eval.metrics import r2_score


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
    answer_acc_threshold: float = 0.80
    ridge: float = 1e-3


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
    d_conf_r2_threshold: float
    answer_acc_threshold: float
    train_seed_base: int
    test_seed_base: int
    d_conf_passed: bool
    answer_passed: bool
    passed: bool


class BypassProbeConfigError(ValueError):
    pass


def collect_bypass_dataset(stream: StreamLike, config: BypassProbeConfig) -> BypassDataset:
    _validate_config(config)
    train_samples = [
        stream.generate(seed=config.train_seed_base + i)
        for i in range(config.n_train_streams)
    ]
    test_samples = [
        stream.generate(seed=config.test_seed_base + i)
        for i in range(config.n_test_streams)
    ]
    samples = train_samples + test_samples
    token_features = []
    targets = []
    train_position_mask = []
    answer_features = []
    answers = []
    train_stream_mask = []
    for sample_idx, sample in enumerate(samples):
        natural_len = int(sample.metadata["natural_len"])
        is_train_stream = sample_idx < config.n_train_streams
        answer_features.append(_sequence_features(sample, stream.vocab_size))
        answers.append(int(sample.answer))
        train_stream_mask.append(is_train_stream)
        for t in range(1, natural_len):
            token_features.append(_window_features(sample.tokens, stream.vocab_size, config.window, t))
            targets.append(float(sample.d_conf[t]))
            train_position_mask.append(is_train_stream)
    return BypassDataset(
        token_features=torch.stack(token_features),
        d_conf=torch.as_tensor(targets, dtype=torch.float32),
        train_position_mask=torch.as_tensor(train_position_mask, dtype=torch.bool),
        answer_features=torch.stack(answer_features),
        answers=torch.as_tensor(answers, dtype=torch.long),
        train_stream_mask=torch.as_tensor(train_stream_mask, dtype=torch.bool),
        n_train_streams=config.n_train_streams,
        n_test_streams=config.n_test_streams,
        train_seed_base=config.train_seed_base,
        test_seed_base=config.test_seed_base,
        num_classes=stream.K,
    )


def run_bypass_probe(
    stream: StreamLike,
    config: BypassProbeConfig | None = None,
    device: str = "cpu",
) -> BypassProbeResult:
    cfg = BypassProbeConfig() if config is None else config
    dataset = collect_bypass_dataset(stream, cfg)
    pred_d_conf = _ridge_predict(dataset, cfg)
    test_mask = ~dataset.train_position_mask
    truth = dataset.d_conf[test_mask]
    pred = pred_d_conf[test_mask]
    d_conf_mse = float(F.mse_loss(pred, truth).item())
    d_conf_r2 = r2_score(pred, truth)
    answer_accuracy = _answer_accuracy(dataset, cfg, device)
    d_conf_passed = d_conf_r2 <= cfg.d_conf_r2_threshold
    answer_passed = answer_accuracy <= cfg.answer_acc_threshold
    return BypassProbeResult(
        d_conf_mse=d_conf_mse,
        d_conf_r2=d_conf_r2,
        answer_accuracy=answer_accuracy,
        n_train_positions=int(dataset.train_position_mask.sum().item()),
        n_test_positions=int(test_mask.sum().item()),
        n_train_streams=cfg.n_train_streams,
        n_test_streams=cfg.n_test_streams,
        window=cfg.window,
        d_conf_r2_threshold=cfg.d_conf_r2_threshold,
        answer_acc_threshold=cfg.answer_acc_threshold,
        train_seed_base=cfg.train_seed_base,
        test_seed_base=cfg.test_seed_base,
        d_conf_passed=d_conf_passed,
        answer_passed=answer_passed,
        passed=d_conf_passed and answer_passed,
    )


def _validate_config(config: BypassProbeConfig) -> None:
    if config.n_train_streams < 1 or config.n_test_streams < 1:
        msg = "bypass probe needs at least one train and one test stream"
        raise BypassProbeConfigError(msg)
    if config.window < 1:
        msg = "bypass probe window must be positive"
        raise BypassProbeConfigError(msg)
    train_start = config.train_seed_base
    train_stop = config.train_seed_base + config.n_train_streams
    test_start = config.test_seed_base
    test_stop = config.test_seed_base + config.n_test_streams
    if max(train_start, test_start) < min(train_stop, test_stop):
        msg = "train and test seed ranges must be disjoint"
        raise BypassProbeConfigError(msg)


def _window_features(
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


def _sequence_features(sample: StreamSample, vocab_size: int) -> torch.Tensor:
    natural_len = int(sample.metadata["natural_len"])
    feature = torch.zeros(sample.tokens.shape[0], vocab_size, dtype=torch.float32)
    tokens = torch.as_tensor(sample.tokens[:natural_len], dtype=torch.long)
    for idx, tok in enumerate(tokens.tolist()):
        feature[idx, int(tok)] = 1.0
    return feature.reshape(-1)


def _ridge_predict(dataset: BypassDataset, config: BypassProbeConfig) -> torch.Tensor:
    x = _with_bias(dataset.token_features.to(torch.float64))
    y = dataset.d_conf.to(torch.float64).unsqueeze(1)
    x_train = x[dataset.train_position_mask]
    y_train = y[dataset.train_position_mask]
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(
        x_train.T @ x_train + config.ridge * eye,
        x_train.T @ y_train,
    )
    return (x @ weights).squeeze(1).to(torch.float32)


def _with_bias(features: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
    return torch.cat([features, ones], dim=1)


def _answer_accuracy(
    dataset: BypassDataset,
    config: BypassProbeConfig,
    device: str,
) -> float:
    torch.manual_seed(0)
    features = dataset.answer_features.to(device=device, dtype=torch.float32)
    answers = dataset.answers.to(device=device)
    is_train = dataset.train_stream_mask.to(device=device)
    classifier = nn.Linear(features.shape[1], dataset.num_classes).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=5e-2, weight_decay=1e-4)
    for _ in range(config.probe_epochs):
        logits = classifier(features[is_train])
        loss = F.cross_entropy(logits, answers[is_train])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = classifier(features[~is_train]).argmax(dim=1)
        return float((pred == answers[~is_train]).to(torch.float32).mean().item())
