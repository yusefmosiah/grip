from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import torch
import torch.nn.functional as F

from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, make_batch
from grip.models import ContentSparseTransformer, DenseTransformer


BatchTensors: TypeAlias = Mapping[str, torch.Tensor]
TrainRecord = Mapping[str, int | float | str | Mapping[str, float]]


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
    return tuple(
        TrainingTokenBatch(
            seed=step_seed,
            tokens=training_tokens(
                task=request.task,
                seq_len=request.seq_len,
                vocab_size=request.vocab_size,
                n_hypotheses=request.n_hypotheses,
                batch_size=request.batch_size,
                seed=step_seed,
                device=request.device,
            ),
        )
        for step_seed in range(request.seed, request.seed + request.steps)
    )


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
        out = model(batch.tokens)
        loss = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, config.vocab_size),
            batch.tokens[:, 1:].reshape(-1),
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
