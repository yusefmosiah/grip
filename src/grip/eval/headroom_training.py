from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, make_batch
from grip.models import ContentSparseTransformer, DenseTransformer


TrainRecord = Mapping[str, int | float | str | Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class TrainingDataError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


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
    stream = _stream(task, seq_len, vocab_size, n_hypotheses, seed)
    return make_batch(stream, n=batch_size, seed=seed, device=device)["tokens"]


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
    tokens: torch.Tensor,
    steps: int,
    lr: float,
    vocab_size: int,
) -> tuple[TrainRecord, ...]:
    records: list[TrainRecord] = []
    if steps == 0:
        records.append(_train_record(step=0, loss=0.0, lr=lr, tokens=0, event="dry_run"))
        return tuple(records)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        out = model(tokens)
        loss = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        records.append(
            _train_record(
                step=step,
                loss=float(loss.item()),
                lr=lr,
                tokens=int(tokens.numel()),
                event="train",
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
) -> TrainRecord:
    return {
        "event": event,
        "loss": {"total": loss},
        "lr": lr,
        "step": step,
        "tokens": tokens,
    }
