from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .headroom import MRegimeConfig, run_m_regime_smoke
from .headroom_baselines import BASELINE_NAMES
from .headroom_training import TRAINING_SEED_STRIDE
from .noise_floor import MIN_NOISE_FLOOR_SEEDS, load_noise_floor
from .noise_floor_artifact import noise_floor_payload
from .noise_floor_calibration_types import NoiseFloorCalibrationConfig
from .experiment_config import shared_config_kwargs
from .score import score_run
from .score_types import JsonValue


RIGHT_CALIBRATION_SEED_OFFSET = 1_000_000_000


@dataclass(frozen=True, slots=True)
class NoiseFloorCalibrationError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


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
    pairs: list[dict[str, JsonValue]] = []
    for seed in config.seed_ids:
        right_seed = _right_calibration_seed(seed)
        left_dir = config.out_dir / "pairs" / f"seed-{seed}" / "left"
        right_dir = config.out_dir / "pairs" / f"seed-{seed}" / "right"
        _run_calibration_pair(config, seed, right_seed, left_dir, right_dir)
        pairs.append(
            {
                "left": str(left_dir),
                "left_seed": seed,
                "right": str(right_dir),
                "right_seed": right_seed,
            }
        )
        for metric in config.metric_names:
            metric_deltas[metric].append(_pair_delta(left_dir, right_dir, metric))
    parsed_deltas = {metric: tuple(float(delta) for delta in deltas) for metric, deltas in metric_deltas.items()}
    metric_ceilings = {
        metric: max(abs(delta) for delta in deltas)
        for metric, deltas in parsed_deltas.items()
    }
    _validate_measurable_spread(metric_ceilings, config.minimum_signal_floor)
    minimum_signal_threshold = dict(metric_ceilings)
    artifact_path = config.out_dir / "noise-floor.json"
    payload = noise_floor_payload(
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
    if len(set(config.decision_seed_ids)) != len(config.decision_seed_ids):
        raise NoiseFloorCalibrationError("decision_seed_ids", "must not contain duplicates")
    calibration_seeds = set(config.seed_ids)
    calibration_seeds.update(_right_calibration_seed(seed) for seed in config.seed_ids)
    if len(calibration_seeds) != len(config.seed_ids) * 2:
        raise NoiseFloorCalibrationError("seed_ids", "left and right calibration seeds must be unique")
    if config.train_steps >= TRAINING_SEED_STRIDE:
        raise NoiseFloorCalibrationError("train_steps", "must be below the per-run seed stride")
    decision_seeds = _decision_seed_space(config)
    if calibration_seeds.intersection(decision_seeds):
        raise NoiseFloorCalibrationError("seed_ids", "must be disjoint from decision seed space")
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
    if config.eval_batch_size <= 0:
        raise NoiseFloorCalibrationError("eval_batch_size", "must be positive")
    if config.eval_seed_offset <= 0:
        raise NoiseFloorCalibrationError("eval_seed_offset", "must be positive")
    if config.eval_seed_offset <= config.train_steps:
        raise NoiseFloorCalibrationError("eval_seed_offset", "must exceed train_steps")
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
    left_seed: int,
    right_seed: int,
    left_dir: Path,
    right_dir: Path,
) -> None:
    run_m_regime_smoke(_m_regime_config(config, left_seed, left_dir))
    run_m_regime_smoke(_m_regime_config(config, right_seed, right_dir))


def _right_calibration_seed(left_seed: int) -> int:
    return left_seed + RIGHT_CALIBRATION_SEED_OFFSET


def _decision_seed_space(config: NoiseFloorCalibrationConfig) -> set[int]:
    seeds = set(config.decision_seed_ids)
    seeds.update(seed + config.eval_seed_offset for seed in config.decision_seed_ids)
    for seed in config.decision_seed_ids:
        start = seed * TRAINING_SEED_STRIDE
        seeds.update(range(start, start + config.train_steps))
    return seeds


def _m_regime_config(
    config: NoiseFloorCalibrationConfig,
    seed: int,
    out_dir: Path,
) -> MRegimeConfig:
    return MRegimeConfig(
        out_dir=out_dir,
        seed=seed,
        **shared_config_kwargs(config),
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


def _validate_measurable_spread(
    metric_ceilings: Mapping[str, float],
    zero_tolerance: float,
) -> None:
    degenerate = tuple(
        metric
        for metric, ceiling in metric_ceilings.items()
        if ceiling <= zero_tolerance
    )
    if degenerate:
        names = ", ".join(sorted(degenerate))
        raise NoiseFloorCalibrationError(
            "metric_deltas",
            f"metrics have no measurable calibration spread: {names}",
        )


def main(argv: Sequence[str] | None = None) -> int:
    from .noise_floor_calibration_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
