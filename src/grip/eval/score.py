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
from .compute import compute_mismatches, run_compute
from .m_regime_validity import noise_floor_validity, run_validity
from .score_types import (
    ComparisonReport,
    JsonValue,
    NoiseFloorArtifact,
    NoiseFloorError,
    RunScore,
    ScoreArtifactError,
)


BOOKKEEPING_METRICS: Final = frozenset({"tokens"})
DEFAULT_COMPUTE_TOLERANCE: Final = 0.05


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
    return RunScore(run_dir=run_dir, metrics=metrics, compute=run_compute(run_dir))


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
    compute_tolerance: float = DEFAULT_COMPUTE_TOLERANCE,
) -> ComparisonReport:
    if not runs:
        raise ScoreArtifactError(Path("comparison.json"), "at least one run is required")
    if compute_tolerance < 0:
        raise ScoreArtifactError(Path("comparison.json"), "compute_tolerance must be non-negative")
    scores = tuple(score_run(run_dir) for run_dir in runs)
    compute_mismatch_fields = compute_mismatches(scores, compute_tolerance)
    noise_floor: NoiseFloorArtifact | None = None
    interpretable = False
    reason = "noise_floor_missing"
    if noise_floor_path is not None:
        try:
            noise_floor = load_noise_floor(noise_floor_path)
            missing_metrics = _missing_noise_floor_metrics(scores, noise_floor)
            config_mismatches = _noise_floor_config_mismatches(scores, noise_floor)
            validity_failures = _run_validity_failures(scores, noise_floor)
            if config_mismatches:
                reason = "noise_floor_config_mismatch"
            elif validity_failures:
                reason = "below_minimum_validity"
            elif missing_metrics:
                reason = "noise_floor_missing_metric"
            elif compute_mismatch_fields:
                reason = "compute_mismatch"
            else:
                interpretable = preregistered
                reason = "ok" if preregistered else "comparison_not_preregistered"
        except NoiseFloorError:
            reason = "noise_floor_invalid"
            config_mismatches = ()
            validity_failures = ()
    else:
        config_mismatches = ()
        validity_failures = ()
    report = ComparisonReport(
        runs=scores,
        interpretable=interpretable,
        reason=reason,
        noise_floor=noise_floor,
        compute_tolerance=compute_tolerance,
        config_mismatches=config_mismatches,
        compute_mismatches=compute_mismatch_fields,
        validity_failures=validity_failures,
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


def _noise_floor_config_mismatches(
    scores: Sequence[RunScore],
    noise_floor: NoiseFloorArtifact,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for score in scores:
        config_path = score.run_dir / "config.resolved.json"
        try:
            run_config = _load_run_config(config_path)
        except ScoreArtifactError:
            mismatches.append(f"{score.run_dir.name}.config.resolved.json")
            continue
        mismatches.extend(_run_config_mismatches(score.run_dir.name, run_config, noise_floor.calibration))
    return tuple(sorted(set(mismatches)))


def _run_validity_failures(
    scores: Sequence[RunScore],
    noise_floor: NoiseFloorArtifact,
) -> tuple[str, ...]:
    failures: list[str] = list(noise_floor_validity(len(noise_floor.calibration_pairs)))
    for score in scores:
        config_path = score.run_dir / "config.resolved.json"
        try:
            run_config = _load_run_config(config_path)
        except ScoreArtifactError:
            failures.append(f"{score.run_dir.name}.config.resolved.json")
            continue
        spec_failures = run_validity(run_config)
        if spec_failures:
            failures.extend(f"{score.run_dir.name}.{field}" for field in spec_failures)
    return tuple(sorted(set(failures)))


def _load_run_config(path: Path) -> Mapping[str, JsonValue]:
    if not path.exists():
        raise ScoreArtifactError(path, "config.resolved.json is required")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoreArtifactError(path, "config.resolved.json must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ScoreArtifactError(path, "config.resolved.json must be a JSON object")
    return raw


def _run_config_mismatches(
    run_name: str,
    run_config: Mapping[str, JsonValue],
    calibration: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    fields = (
        ("data.task", ("data", "task"), ("data", "task")),
        ("data.seq_len", ("data", "seq_len"), ("data", "seq_len")),
        ("data.vocab_size", ("data", "vocab_size"), ("data", "vocab_size")),
        ("device", ("device",), ("run", "device")),
        ("eval.batch_size", ("eval", "batch_size"), ("eval", "batch_size")),
        ("eval.seed_offset", ("eval", "seed_offset"), ("eval", "seed_offset")),
        ("model.d_model", ("model", "d_model"), ("model", "d_model")),
        ("model.n_heads", ("model", "n_heads"), ("model", "n_heads")),
        ("model.n_hypotheses", ("model", "n_hypotheses"), ("model", "n_hypotheses")),
        ("model.n_layers", ("model", "n_layers"), ("model", "n_layers")),
        ("sparse.block_size", ("sparse", "block_size"), ("sparse", "block_size")),
        ("sparse.top_k_blocks", ("sparse", "top_k_blocks"), ("sparse", "top_k_blocks")),
        ("sparse.window", ("sparse", "window"), ("sparse", "window")),
        ("train.batch_size", ("train", "batch_size"), ("train", "batch_size")),
        ("train.lr", ("train", "lr"), ("train", "lr")),
        ("train.steps", ("train", "steps"), ("train", "steps")),
    )
    mismatches = [
        f"{run_name}.{label}"
        for label, calibration_path, run_path in fields
        if _field_value(calibration, calibration_path) != _field_value(run_config, run_path)
    ]
    baseline_names = _field_value(calibration, ("baseline_names",))
    model_name = _field_value(run_config, ("model", "name"))
    if not isinstance(baseline_names, list) or model_name not in baseline_names:
        mismatches.append(f"{run_name}.baseline_names")
    return tuple(mismatches)


def _field_value(payload: Mapping[str, JsonValue], path: tuple[str, ...]) -> JsonValue:
    current: JsonValue = dict(payload)
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


if __name__ == "__main__":
    raise SystemExit("CODEX: wire CLI args -> compare([Path(a) for a in argv[1:]])")
