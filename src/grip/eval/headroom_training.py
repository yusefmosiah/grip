from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import torch
import torch.nn.functional as F

from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, make_batch
from grip.models import ContentSparseTransformer, DenseTransformer


BatchTensors: TypeAlias = Mapping[str, torch.Tensor]
TrainRecord = Mapping[str, int | float | str | Mapping[str, float]]
TRAINING_SEED_STRIDE = 1_000_000


@dataclass(frozen=True, slots=True)
class TrainingDataError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class TrainingTokenBatch:
    seed: int
    tokens: torch.Tensor
    real_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TrainingBatchRequest:
    task: str
    seq_len: int
    vocab_size: int
    n_hypotheses: int
    batch_size: int
    seed: int
    steps: int
    device: str


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    dry_run_seed: int
    lr: float
    vocab_size: int


def training_tokens(
    *,
    task: str,
    seq_len: int,
    vocab_size: int,
    n_hypotheses: int,
    batch_size: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    return training_batch(
        task=task,
        seq_len=seq_len,
        vocab_size=vocab_size,
        n_hypotheses=n_hypotheses,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )["tokens"]


def training_batch(
    *,
    task: str,
    seq_len: int,
    vocab_size: int,
    n_hypotheses: int,
    batch_size: int,
    seed: int,
    device: str,
) -> BatchTensors:
    stream = _stream(task, seq_len, vocab_size, n_hypotheses, seed)
    return make_batch(stream, n=batch_size, seed=seed, device=device)


def training_token_batches(request: TrainingBatchRequest) -> tuple[TrainingTokenBatch, ...]:
    if request.steps >= TRAINING_SEED_STRIDE:
        raise TrainingDataError("steps", "must be below the per-run seed stride")
    batches: list[TrainingTokenBatch] = []
    for step_seed in _step_seeds(request.seed, request.steps):
        batch = training_batch(
            task=request.task,
            seq_len=request.seq_len,
            vocab_size=request.vocab_size,
            n_hypotheses=request.n_hypotheses,
            batch_size=request.batch_size,
            seed=step_seed,
            device=request.device,
        )
        batches.append(
            TrainingTokenBatch(
                seed=step_seed,
                tokens=batch["tokens"],
                real_mask=batch["real_mask"],
            )
        )
    return tuple(batches)


def _step_seeds(run_seed: int, steps: int) -> range:
    start = run_seed * TRAINING_SEED_STRIDE
    return range(start, start + steps)


def _stream(
    task: str,
    seq_len: int,
    vocab_size: int,
    n_hypotheses: int,
    seed: int,
) -> BayesianEvidenceStream | SourceReliabilityReversalStream:
    if task == "bayesian":
        return BayesianEvidenceStream(
            num_hypotheses=n_hypotheses,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed,
        )
    if task == "reversal":
        return SourceReliabilityReversalStream(seq_len=seq_len, seed=seed)
    raise TrainingDataError("task", "must be bayesian or reversal")


def train_model(
    *,
    model: DenseTransformer | ContentSparseTransformer,
    batches: tuple[TrainingTokenBatch, ...],
    config: TrainingLoopConfig,
) -> tuple[TrainRecord, ...]:
    records: list[TrainRecord] = []
    if not batches:
        records.append(
            _train_record(
                step=0,
                loss=0.0,
                lr=config.lr,
                tokens=0,
                event="dry_run",
                seed=config.dry_run_seed,
            )
        )
        return tuple(records)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for step, batch in enumerate(batches, start=1):
        out = model(batch.tokens, real_mask=batch.real_mask)
        loss = next_token_loss(
            logits=out["lm_logits"],
            tokens=batch.tokens,
            real_mask=batch.real_mask,
            vocab_size=config.vocab_size,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        records.append(
            _train_record(
                step=step,
                loss=float(loss.item()),
                lr=config.lr,
                tokens=int(batch.tokens.numel()),
                event="train",
                seed=batch.seed,
            )
        )
    return tuple(records)


def next_token_loss(
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    real_mask: torch.Tensor | None,
    vocab_size: int,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(tokens[:, 1:])
    if real_mask is None:
        return per_token.mean()
    target_mask = real_mask[:, 1:].to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * target_mask).sum() / target_mask.sum().clamp_min(1.0)


def _train_record(
    *,
    step: int,
    loss: float,
    lr: float,
    tokens: int,
    event: str,
    seed: int,
) -> TrainRecord:
    return {
        "event": event,
        "loss": {"total": loss},
        "lr": lr,
        "seed": seed,
        "step": step,
        "tokens": tokens,
    }
