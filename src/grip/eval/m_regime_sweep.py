from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypeAlias

from .aggregate_summary import AggregateSummaryResult, aggregate_summary_file
from .headroom import MRegimeConfig, MRegimeResult, run_m_regime_smoke
from .noise_floor import MIN_NOISE_FLOOR_SEEDS
from .noise_floor_calibration import NoiseFloorCalibrationConfig, calibrate_noise_floor


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class MRegimeSweepConfig:
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


@dataclass(frozen=True, slots=True)
class MRegimeSweepResult:
    noise_floor_path: Path
    summary_path: Path
    aggregate: AggregateSummaryResult


def run_m_regime_sweep(config: MRegimeSweepConfig) -> MRegimeSweepResult:
    noise_floor = calibrate_noise_floor(_calibration_config(config))
    rows = tuple(_run_seed(config, noise_floor.path, seed) for seed in config.seed_ids)
    summary_path = config.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_summary_payload(config, noise_floor.path, rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = aggregate_summary_file(summary_path, config.out_dir / "aggregate")
    return MRegimeSweepResult(
        noise_floor_path=noise_floor.path,
        summary_path=summary_path,
        aggregate=aggregate,
    )


def _calibration_config(config: MRegimeSweepConfig) -> NoiseFloorCalibrationConfig:
    return NoiseFloorCalibrationConfig(
        out_dir=config.out_dir / "noise-floor",
        task=config.task,
        seed_ids=config.seed_ids,
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


def _run_seed(config: MRegimeSweepConfig, noise_floor_path: Path, seed: int) -> dict[str, JsonValue]:
    result = run_m_regime_smoke(_headroom_config(config, noise_floor_path, seed))
    losses = _losses(result)
    return {
        "authorize_avsb": result.authorize_avsb,
        "content_minus_dense": losses.content_sparse - losses.dense,
        "content_sparse": losses.content_sparse,
        "dense": losses.dense,
        "interpretable": result.comparison.interpretable,
        "local": losses.local,
        "reason": result.comparison.reason,
        "seed": seed,
        "status": result.status,
    }


def _headroom_config(config: MRegimeSweepConfig, noise_floor_path: Path, seed: int) -> MRegimeConfig:
    return MRegimeConfig(
        out_dir=config.out_dir / "decisions" / f"seed-{seed}",
        noise_floor_path=noise_floor_path,
        preregistered=True,
        task=config.task,
        device=config.device,
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
    )


@dataclass(frozen=True, slots=True)
class _Losses:
    dense: float
    local: float
    content_sparse: float


def _losses(result: MRegimeResult) -> _Losses:
    by_name = {score.run_dir.name: score.metrics["loss"] for score in result.comparison.runs}
    return _Losses(
        dense=by_name["dense"],
        local=by_name["local"],
        content_sparse=by_name["content-sparse"],
    )


def _summary_payload(
    config: MRegimeSweepConfig,
    noise_floor_path: Path,
    rows: tuple[dict[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return {
        config.task: {
            "floor": {"path": str(noise_floor_path)},
            "rows": list(rows),
        },
    }
