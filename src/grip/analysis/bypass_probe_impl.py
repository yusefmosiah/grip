from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from grip.eval.metrics import r2_score

from .bypass_helpers import (
    answer_leak_features,
    candidate_ridges,
    candidate_windows,
    validate_config,
    window_features,
)
from .bypass_types import (
    BypassDataset,
    BypassProbeConfig,
    BypassProbeConfigError,
    BypassProbeResult,
    StreamLike,
)


def collect_bypass_dataset(stream: StreamLike, config: BypassProbeConfig) -> BypassDataset:
    validate_config(config)
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
        answer_features.append(answer_leak_features(sample, stream.vocab_size, config.window))
        answers.append(int(sample.answer))
        train_stream_mask.append(is_train_stream)
        for t in range(1, natural_len):
            token_features.append(window_features(sample.tokens, stream.vocab_size, config.window, t))
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
        vocab_size=stream.vocab_size,
    )


def run_bypass_probe(
    stream: StreamLike,
    config: BypassProbeConfig | None = None,
    device: str = "cpu",
) -> BypassProbeResult:
    cfg = BypassProbeConfig() if config is None else config
    best_d_conf = _best_d_conf_probe(stream, cfg)
    answer_result = _best_answer_probe(stream, cfg, device)
    dataset = best_d_conf.dataset
    test_mask = ~dataset.train_position_mask
    truth = dataset.d_conf[test_mask]
    pred = best_d_conf.prediction[test_mask]
    d_conf_mse = float(F.mse_loss(pred, truth).item())
    answer_acc_threshold = _answer_accuracy_threshold(dataset.num_classes, cfg)
    d_conf_passed = best_d_conf.r2 <= cfg.d_conf_r2_threshold
    answer_passed = answer_result.accuracy <= answer_acc_threshold
    positive_control_passed = best_d_conf.positive_control_r2 >= cfg.positive_control_r2_threshold
    return BypassProbeResult(
        d_conf_mse=d_conf_mse,
        d_conf_r2=best_d_conf.r2,
        answer_accuracy=answer_result.accuracy,
        n_train_positions=int(dataset.train_position_mask.sum().item()),
        n_test_positions=int(test_mask.sum().item()),
        n_train_streams=cfg.n_train_streams,
        n_test_streams=cfg.n_test_streams,
        window=best_d_conf.window,
        ridge=best_d_conf.ridge,
        answer_window=answer_result.window,
        window_grid=candidate_windows(cfg),
        ridge_grid=candidate_ridges(cfg),
        d_conf_r2_threshold=cfg.d_conf_r2_threshold,
        answer_acc_threshold=answer_acc_threshold,
        positive_control_r2=best_d_conf.positive_control_r2,
        positive_control_r2_threshold=cfg.positive_control_r2_threshold,
        positive_control_passed=positive_control_passed,
        answer_train_loss_initial=answer_result.train_loss_initial,
        answer_train_loss_final=answer_result.train_loss_final,
        answer_converged=answer_result.converged,
        train_seed_base=cfg.train_seed_base,
        test_seed_base=cfg.test_seed_base,
        d_conf_passed=d_conf_passed,
        answer_passed=answer_passed,
        passed=d_conf_passed and answer_passed and positive_control_passed and answer_result.converged,
    )


def _answer_accuracy_threshold(num_classes: int, config: BypassProbeConfig) -> float:
    if config.answer_acc_threshold is not None:
        return config.answer_acc_threshold
    chance = 1.0 / float(num_classes)
    return min(1.0, config.answer_acc_chance_multiplier * chance)


def _ridge_predict(dataset: BypassDataset, config: BypassProbeConfig) -> torch.Tensor:
    x = _with_bias(dataset.token_features.to(torch.float64))
    y = dataset.d_conf.to(torch.float64).unsqueeze(1)
    x_train = x[dataset.train_position_mask]
    y_train = y[dataset.train_position_mask]
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(x_train.T @ x_train + config.ridge * eye, x_train.T @ y_train)
    return (x @ weights).squeeze(1).to(torch.float32)


@dataclass(frozen=True, slots=True)
class _DConfProbeResult:
    dataset: BypassDataset
    prediction: torch.Tensor
    r2: float
    positive_control_r2: float
    window: int
    ridge: float


def _best_d_conf_probe(stream: StreamLike, config: BypassProbeConfig) -> _DConfProbeResult:
    best: _DConfProbeResult | None = None
    for window in candidate_windows(config):
        dataset = collect_bypass_dataset(stream, replace(config, window=window))
        for ridge in candidate_ridges(config):
            cell_config = replace(config, window=window, ridge=ridge)
            prediction = _ridge_predict(dataset, cell_config)
            test_mask = ~dataset.train_position_mask
            score = r2_score(prediction[test_mask], dataset.d_conf[test_mask])
            positive_score = _positive_control_r2(dataset, cell_config)
            if best is None or score > best.r2:
                best = _DConfProbeResult(dataset, prediction, score, positive_score, window, ridge)
    if best is None:
        raise BypassProbeConfigError("bypass probe grid must contain at least one cell")
    return best


def _positive_control_r2(dataset: BypassDataset, config: BypassProbeConfig) -> float:
    target = _positive_control_target(dataset, config)
    if target is None:
        return 0.0
    control_dataset = replace(dataset, d_conf=target)
    prediction = _ridge_predict(control_dataset, config)
    test_mask = ~dataset.train_position_mask
    return r2_score(prediction[test_mask], target[test_mask])


def _positive_control_target(dataset: BypassDataset, config: BypassProbeConfig) -> torch.Tensor | None:
    train_mask = dataset.train_position_mask
    test_mask = ~train_mask
    features = dataset.token_features.reshape(-1, config.window, dataset.vocab_size)
    token_ids = features[:, -1, :].argmax(dim=1).to(torch.float32)
    if token_ids[train_mask].var(unbiased=False) <= 0.0 or token_ids[test_mask].var(unbiased=False) <= 0.0:
        return None
    return token_ids / max(1.0, float(dataset.vocab_size - 1))


def _with_bias(features: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
    return torch.cat([features, ones], dim=1)


@dataclass(frozen=True, slots=True)
class _AnswerProbeResult:
    accuracy: float
    train_loss_initial: float
    train_loss_final: float
    converged: bool
    window: int


def _best_answer_probe(stream: StreamLike, config: BypassProbeConfig, device: str) -> _AnswerProbeResult:
    best: _AnswerProbeResult | None = None
    for window in candidate_windows(config):
        dataset = collect_bypass_dataset(stream, replace(config, window=window))
        result = _answer_accuracy(dataset, config, device, window)
        if best is None or result.accuracy > best.accuracy:
            best = result
    if best is None:
        raise BypassProbeConfigError("bypass probe grid must contain at least one window")
    return best


def _answer_accuracy(
    dataset: BypassDataset,
    config: BypassProbeConfig,
    device: str,
    window: int,
) -> _AnswerProbeResult:
    features = dataset.answer_features.to(device=device, dtype=torch.float32)
    answers = dataset.answers.to(device=device)
    is_train = dataset.train_stream_mask.to(device=device)
    classifier = nn.Linear(features.shape[1], dataset.num_classes).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=5e-2, weight_decay=1e-4)
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(classifier(features[is_train]), answers[is_train]).item())
    for _ in range(config.probe_epochs):
        loss = F.cross_entropy(classifier(features[is_train]), answers[is_train])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_loss = float(F.cross_entropy(classifier(features[is_train]), answers[is_train]).item())
        pred = classifier(features[~is_train]).argmax(dim=1)
        accuracy = float((pred == answers[~is_train]).to(torch.float32).mean().item())
    return _AnswerProbeResult(
        accuracy=accuracy,
        train_loss_initial=initial_loss,
        train_loss_final=final_loss,
        converged=final_loss <= initial_loss - config.answer_convergence_min_delta,
        window=window,
    )
