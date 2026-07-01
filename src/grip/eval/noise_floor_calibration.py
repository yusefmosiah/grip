from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

from .headroom import MRegimeConfig, run_m_regime_smoke
from .noise_floor import MIN_NOISE_FLOOR_SEEDS, load_noise_floor
from .score import score_run


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
BASELINE_NAMES: tuple[str, ...] = ("dense", "local", "content-sparse")


@dataclass(frozen=True, slots=True)
class NoiseFloorCalibrationError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class NoiseFloorCalibrationConfig:
    out_dir: Path
    task: str = "bayesian"
    seed_ids: tuple[int, ...] = tuple(range(MIN_NOISE_FLOOR_SEEDS))
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
    lr: float = 1e-3
    device: str = "cpu"
    metric_names: tuple[str, ...] = ("loss",)
    minimum_signal_floor: float = 1e-6


@dataclass(frozen=True, slots=True)
class NoiseFloorCalibrationResult:
    path: Path
    metric_deltas: Mapping[str, tuple[float, ...]]
    metric_ceilings: Mapping[str, float]
    minimum_signal_threshold: Mapping[str, float]


def calibrate_noise_floor(config: NoiseFloorCalibrationConfig) -> NoiseFloorCalibrationResult:
    _validate_calibration_config(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    metric_deltas = {metric: [] for metric in config.metric_names}
    pairs: list[dict[str, str]] = []
    for seed in config.seed_ids:
        left_dir = config.out_dir / "pairs" / f"seed-{seed}" / "left"
        right_dir = config.out_dir / "pairs" / f"seed-{seed}" / "right"
        _run_calibration_pair(config, seed, left_dir, right_dir)
        pairs.append({"left": str(left_dir), "right": str(right_dir)})
        for metric in config.metric_names:
            metric_deltas[metric].append(_pair_delta(left_dir, right_dir, metric))
    parsed_deltas = {
        metric: tuple(float(delta) for delta in deltas)
        for metric, deltas in metric_deltas.items()
    }
    metric_ceilings = {
        metric: max(abs(delta) for delta in deltas)
        for metric, deltas in parsed_deltas.items()
    }
    minimum_signal_threshold = {
        metric: max(metric_ceilings[metric], config.minimum_signal_floor)
        for metric in config.metric_names
    }
    artifact_path = config.out_dir / "noise-floor.json"
    payload = _artifact_payload(
        config,
        pairs,
        parsed_deltas,
        metric_ceilings,
        minimum_signal_threshold,
    )
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    load_noise_floor(artifact_path)
    return NoiseFloorCalibrationResult(
        path=artifact_path,
        metric_deltas=parsed_deltas,
        metric_ceilings=metric_ceilings,
        minimum_signal_threshold=minimum_signal_threshold,
    )


def _validate_calibration_config(config: NoiseFloorCalibrationConfig) -> None:
    if len(config.seed_ids) < MIN_NOISE_FLOOR_SEEDS:
        raise NoiseFloorCalibrationError("seed_ids", "must contain at least 8 seeds")
    if len(set(config.seed_ids)) != len(config.seed_ids):
        raise NoiseFloorCalibrationError("seed_ids", "must not contain duplicates")
    if not config.metric_names:
        raise NoiseFloorCalibrationError("metric_names", "must not be empty")
    if len(set(config.metric_names)) != len(config.metric_names):
        raise NoiseFloorCalibrationError("metric_names", "must not contain duplicates")
    if config.task not in {"bayesian", "reversal"}:
        raise NoiseFloorCalibrationError("task", "must be bayesian or reversal")
    if config.train_steps < 0:
        raise NoiseFloorCalibrationError("train_steps", "must be non-negative")
    if config.train_batch_size <= 0:
        raise NoiseFloorCalibrationError("train_batch_size", "must be positive")
    if config.lr <= 0:
        raise NoiseFloorCalibrationError("lr", "must be positive")
    if config.device != "cpu":
        raise NoiseFloorCalibrationError("device", "must be cpu for calibration")
    if config.minimum_signal_floor < 0:
        raise NoiseFloorCalibrationError("minimum_signal_floor", "must be non-negative")
    if config.task == "reversal" and config.seq_len < 16:
        raise NoiseFloorCalibrationError("seq_len", "must be >= 16 for reversal")
    if config.task == "reversal" and config.vocab_size != 64:
        raise NoiseFloorCalibrationError("vocab_size", "must be 64 for reversal")
    if config.task == "reversal" and config.n_hypotheses != 4:
        raise NoiseFloorCalibrationError("n_hypotheses", "must be 4 for reversal")


def _run_calibration_pair(
    config: NoiseFloorCalibrationConfig,
    seed: int,
    left_dir: Path,
    right_dir: Path,
) -> None:
    run_m_regime_smoke(_m_regime_config(config, seed, left_dir))
    run_m_regime_smoke(_m_regime_config(config, seed, right_dir))


def _m_regime_config(
    config: NoiseFloorCalibrationConfig,
    seed: int,
    out_dir: Path,
) -> MRegimeConfig:
    return MRegimeConfig(
        out_dir=out_dir,
        task=config.task,
        seed=seed,
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
        lr=config.lr,
        device=config.device,
    )


def _pair_delta(left_dir: Path, right_dir: Path, metric: str) -> float:
    deltas = tuple(
        _metric_delta(left_dir / baseline, right_dir / baseline, metric)
        for baseline in BASELINE_NAMES
    )
    return max(deltas, key=abs)


def _metric_delta(left_run_dir: Path, right_run_dir: Path, metric: str) -> float:
    left_score = score_run(left_run_dir)
    right_score = score_run(right_run_dir)
    if metric not in left_score.metrics or metric not in right_score.metrics:
        raise NoiseFloorCalibrationError("metric_names", f"metric {metric!r} is missing")
    return right_score.metrics[metric] - left_score.metrics[metric]


def _artifact_payload(
    config: NoiseFloorCalibrationConfig,
    pairs: Sequence[dict[str, str]],
    metric_deltas: dict[str, tuple[float, ...]],
    metric_ceilings: dict[str, float],
    minimum_signal_threshold: dict[str, float],
) -> dict[str, JsonValue]:
    return {
        "calibration": {
            "baseline_names": list(BASELINE_NAMES),
            "device": config.device,
            "minimum_signal_floor": config.minimum_signal_floor,
            "task": config.task,
            "train_batch_size": config.train_batch_size,
            "train_steps": config.train_steps,
        },
        "identical_config_pairs": list(pairs),
        "kind": "M-noise-floor",
        "metric_ceilings": metric_ceilings,
        "metric_deltas": {
            metric: list(deltas)
            for metric, deltas in metric_deltas.items()
        },
        "minimum_signal_threshold": minimum_signal_threshold,
        "seed_count": len(config.seed_ids),
        "seed_ids": list(config.seed_ids),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--task", choices=("bayesian", "reversal"), default="bayesian")
    parser.add_argument("--seed-count", type=int, default=MIN_NOISE_FLOOR_SEEDS)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=17)
    parser.add_argument("--n-hypotheses", type=int, default=3)
    args = parser.parse_args(argv)
    result = calibrate_noise_floor(
        NoiseFloorCalibrationConfig(
            out_dir=args.out_dir,
            task=args.task,
            seed_ids=tuple(range(args.seed_count)),
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            n_hypotheses=args.n_hypotheses,
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
        )
    )
    print(result.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
