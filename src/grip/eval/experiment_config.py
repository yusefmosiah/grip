from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .score_types import JsonValue


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    task: str = "bayesian"
    seq_len: int = 8
    vocab_size: int = 17
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 1
    n_hypotheses: int = 3
    block_size: int = 2
    top_k_blocks: int = 3
    window: int = 2
    train_steps: int = 0
    train_batch_size: int = 1
    eval_batch_size: int = 1
    eval_seed_offset: int = 10_000
    lr: float = 1e-3
    device: str = "cpu"


class ExperimentConfigLike(Protocol):
    task: str
    seq_len: int
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    n_hypotheses: int
    block_size: int
    top_k_blocks: int
    window: int
    train_steps: int
    train_batch_size: int
    eval_batch_size: int
    eval_seed_offset: int
    lr: float
    device: str


def experiment_config(config: ExperimentConfigLike) -> ExperimentConfig:
    return ExperimentConfig(
        task=config.task,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_hypotheses=config.n_hypotheses,
        block_size=config.block_size,
        top_k_blocks=config.top_k_blocks,
        window=config.window,
        train_steps=config.train_steps,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        eval_seed_offset=config.eval_seed_offset,
        lr=config.lr,
        device=config.device,
    )


def shared_config_kwargs(config: ExperimentConfigLike) -> dict[str, JsonValue]:
    shared = experiment_config(config)
    return {
        "task": shared.task,
        "seq_len": shared.seq_len,
        "vocab_size": shared.vocab_size,
        "d_model": shared.d_model,
        "n_heads": shared.n_heads,
        "n_layers": shared.n_layers,
        "n_hypotheses": shared.n_hypotheses,
        "block_size": shared.block_size,
        "top_k_blocks": shared.top_k_blocks,
        "window": shared.window,
        "train_steps": shared.train_steps,
        "train_batch_size": shared.train_batch_size,
        "eval_batch_size": shared.eval_batch_size,
        "eval_seed_offset": shared.eval_seed_offset,
        "lr": shared.lr,
        "device": shared.device,
    }


def experiment_provenance_payload(
    config: ExperimentConfigLike,
    *,
    decision_seed_count: int,
    include_device: bool,
) -> dict[str, JsonValue]:
    shared = experiment_config(config)
    payload: dict[str, JsonValue] = {
        "data": {
            "seq_len": shared.seq_len,
            "task": shared.task,
            "vocab_size": shared.vocab_size,
        },
        "decision": {
            "seed_count": decision_seed_count,
        },
        "eval": {
            "batch_size": shared.eval_batch_size,
            "seed_offset": shared.eval_seed_offset,
        },
        "model": {
            "d_model": shared.d_model,
            "n_heads": shared.n_heads,
            "n_hypotheses": shared.n_hypotheses,
            "n_layers": shared.n_layers,
        },
        "sparse": {
            "block_size": shared.block_size,
            "top_k_blocks": shared.top_k_blocks,
            "window": shared.window,
        },
        "train": {
            "batch_size": shared.train_batch_size,
            "lr": shared.lr,
            "steps": shared.train_steps,
        },
    }
    if include_device:
        payload["device"] = shared.device
    return payload
