"""Score one or more runs and emit the comparison. Separate from training.

Usage: python -m grip.eval.score run_dir1 run_dir2 ...
Reads each run's eval tensors + config, computes all metrics, writes a
comparison JSON. This is where 'who won' is decided — never in the trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Mapping, Sequence

from .noise_floor import is_number, load_noise_floor
from .score_types import (
    ComparisonReport,
    NoiseFloorArtifact,
    NoiseFloorError,
    RunScore,
    ScoreArtifactError,
)


BOOKKEEPING_METRICS: Final = frozenset({"tokens"})


def score_run(run_dir: Path) -> RunScore:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists() and (run_dir / "eval_tensors.json").exists():
        write_metrics(run_dir, _metrics_from_eval_artifact(run_dir / "eval_tensors.json"))
    if not metrics_path.exists():
        raise ScoreArtifactError(metrics_path, "metrics.json is required")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScoreArtifactError(metrics_path, "metrics must be a JSON object")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise ScoreArtifactError(metrics_path, "metric names must be strings")
        if not is_number(value):
            raise ScoreArtifactError(metrics_path, f"metric {name!r} must be numeric")
        metrics[name] = float(value)
    return RunScore(run_dir=run_dir, metrics=metrics)


def write_metrics(run_dir: Path, metrics: Mapping[str, int | float]) -> Path:
    parsed: dict[str, float] = {}
    for name, value in metrics.items():
        if not is_number(value):
            raise ScoreArtifactError(run_dir / "metrics.json", f"metric {name!r} must be numeric")
        parsed[name] = float(value)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _metrics_from_eval_artifact(path: Path) -> Mapping[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScoreArtifactError(path, "eval_tensors must be a JSON object")
    loss = raw.get("loss")
    tokens = raw.get("tokens")
    if not is_number(loss):
        raise ScoreArtifactError(path, "loss must be numeric")
    if not is_number(tokens):
        raise ScoreArtifactError(path, "tokens must be numeric")
    return {"loss": float(loss), "tokens": float(tokens)}


def compare(
    runs: Sequence[Path],
    noise_floor_path: Path | None = None,
    *,
    preregistered: bool = False,
) -> ComparisonReport:
    if not runs:
        raise ScoreArtifactError(Path("comparison.json"), "at least one run is required")
    scores = tuple(score_run(run_dir) for run_dir in runs)
    noise_floor: NoiseFloorArtifact | None = None
    interpretable = False
    reason = "noise_floor_missing"
    if noise_floor_path is not None:
        try:
            noise_floor = load_noise_floor(noise_floor_path)
            missing_metrics = _missing_noise_floor_metrics(scores, noise_floor)
            if missing_metrics:
                reason = "noise_floor_missing_metric"
            else:
                interpretable = preregistered
                reason = "ok" if preregistered else "comparison_not_preregistered"
        except NoiseFloorError:
            reason = "noise_floor_invalid"
    report = ComparisonReport(
        runs=scores,
        interpretable=interpretable,
        reason=reason,
        noise_floor=noise_floor,
    )
    comparison_path = runs[0].parent / "comparison.json"
    comparison_path.write_text(report.to_json_text(), encoding="utf-8")
    return report


def _missing_noise_floor_metrics(
    scores: Sequence[RunScore],
    noise_floor: NoiseFloorArtifact,
) -> tuple[str, ...]:
    compared_metrics = sorted(
        {
            metric
            for score in scores
            for metric in score.metrics
            if metric not in BOOKKEEPING_METRICS
        }
    )
    return tuple(
        metric
        for metric in compared_metrics
        if metric not in noise_floor.minimum_signal_threshold
        or metric not in noise_floor.metric_deltas
        or metric not in noise_floor.metric_ceilings
    )


if __name__ == "__main__":
    raise SystemExit("CODEX: wire CLI args -> compare([Path(a) for a in argv[1:]])")
