from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .noise_floor import MIN_NOISE_FLOOR_SEEDS


DEFAULT_CALIBRATION_SEED_OFFSET = 10_000_000_000


@dataclass(frozen=True, slots=True)
class NoiseFloorCalibrationConfig:
    out_dir: Path
    task: str = "bayesian"
    seed_ids: tuple[int, ...] = tuple(
        DEFAULT_CALIBRATION_SEED_OFFSET + seed
        for seed in range(MIN_NOISE_FLOOR_SEEDS)
    )
    decision_seed_ids: tuple[int, ...] = tuple(range(MIN_NOISE_FLOOR_SEEDS))
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
    metric_names: tuple[str, ...] = ("loss",)
    minimum_signal_floor: float = 1e-6
