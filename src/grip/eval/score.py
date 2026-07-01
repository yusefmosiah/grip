"""Score one or more runs and emit the comparison. Separate from training.

Usage: python -m grip.eval.score run_dir1 run_dir2 ...
Reads each run's eval tensors + config, computes all metrics, writes a
comparison JSON. This is where 'who won' is decided — never in the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Mapping, Sequence, TypeAlias


MIN_NOISE_FLOOR_SEEDS: Final = 8
JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ScoreArtifactError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class NoiseFloorError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class RunScore:
    run_dir: Path
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class NoiseFloorArtifact:
    path: Path
    seed_count: int
    seed_ids: tuple[int, ...]
    identical_config_pairs: tuple[Mapping[str, str], ...]
    minimum_signal_threshold: Mapping[str, float]
    metric_deltas: Mapping[str, tuple[float, ...]]
    metric_ceilings: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    runs: tuple[RunScore, ...]
    interpretable: bool
    reason: str
    noise_floor: NoiseFloorArtifact | None

    def to_json_text(self) -> str:
        payload = {
            "interpretable": self.interpretable,
            "reason": self.reason,
            "runs": [
                {"run_dir": str(run.run_dir), "metrics": dict(run.metrics)}
                for run in self.runs
            ],
            "noise_floor": None
            if self.noise_floor is None
            else {
                "path": str(self.noise_floor.path),
                "seed_count": self.noise_floor.seed_count,
                "seed_ids": list(self.noise_floor.seed_ids),
                "minimum_signal_threshold": dict(self.noise_floor.minimum_signal_threshold),
                "metric_ceilings": dict(self.noise_floor.metric_ceilings),
            },
        }
        return json.dumps(payload, indent=2)


def score_run(run_dir: Path) -> RunScore:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ScoreArtifactError(metrics_path, "metrics.json is required")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScoreArtifactError(metrics_path, "metrics must be a JSON object")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise ScoreArtifactError(metrics_path, "metric names must be strings")
        if not _is_number(value):
            raise ScoreArtifactError(metrics_path, f"metric {name!r} must be numeric")
        metrics[name] = float(value)
    return RunScore(run_dir=run_dir, metrics=metrics)


def load_noise_floor(path: Path) -> NoiseFloorArtifact:
    if not path.exists():
        raise NoiseFloorError(path, "noise-floor artifact is required")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NoiseFloorError(path, "noise-floor artifact must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise NoiseFloorError(path, "noise-floor artifact must be a JSON object")
    kind = raw.get("kind")
    if kind != "M-noise-floor":
        raise NoiseFloorError(path, "kind must be M-noise-floor")
    seed_count = raw.get("seed_count")
    if not isinstance(seed_count, int) or isinstance(seed_count, bool):
        raise NoiseFloorError(path, "seed_count must be an integer")
    if seed_count < MIN_NOISE_FLOOR_SEEDS:
        raise NoiseFloorError(path, "seed_count must be >= 8")
    seed_ids = _parse_seed_ids(path, raw.get("seed_ids"), seed_count)
    identical_config_pairs = _parse_config_pairs(path, raw.get("identical_config_pairs"), seed_count)
    minimum_signal_threshold = _parse_metric_map(path, "minimum_signal_threshold", raw.get("minimum_signal_threshold"))
    raw_ceilings = raw.get("metric_ceilings")
    metric_ceilings = _parse_metric_map(path, "metric_ceilings", raw_ceilings)
    raw_deltas = raw.get("metric_deltas")
    if not isinstance(raw_deltas, dict):
        raise NoiseFloorError(path, "metric_deltas must be a JSON object")

    metric_deltas: dict[str, tuple[float, ...]] = {}
    for metric_name, deltas in raw_deltas.items():
        if not isinstance(metric_name, str):
            raise NoiseFloorError(path, "metric names must be strings")
        if not isinstance(deltas, list):
            raise NoiseFloorError(path, f"metric {metric_name!r} deltas must be a list")
        parsed = _parse_metric_deltas(path, metric_name, deltas)
        if len(parsed) < seed_count:
            raise NoiseFloorError(path, f"metric {metric_name!r} has fewer deltas than seed_count")
        if metric_name not in metric_ceilings:
            raise NoiseFloorError(path, f"metric {metric_name!r} missing metric ceiling")
        expected_ceiling = max(abs(delta) for delta in parsed)
        if abs(metric_ceilings[metric_name] - expected_ceiling) > 1e-12:
            raise NoiseFloorError(path, f"metric {metric_name!r} ceiling is stale")
        if metric_name not in minimum_signal_threshold:
            raise NoiseFloorError(path, f"metric {metric_name!r} missing minimum signal threshold")
        metric_deltas[metric_name] = parsed
    if not metric_deltas:
        raise NoiseFloorError(path, "at least one metric delta series is required")
    return NoiseFloorArtifact(
        path=path,
        seed_count=seed_count,
        seed_ids=seed_ids,
        identical_config_pairs=identical_config_pairs,
        minimum_signal_threshold=minimum_signal_threshold,
        metric_deltas=metric_deltas,
        metric_ceilings=metric_ceilings,
    )


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


def _parse_metric_deltas(
    path: Path,
    metric_name: str,
    deltas: Sequence[int | float],
) -> tuple[float, ...]:
    parsed: list[float] = []
    for delta in deltas:
        if not _is_number(delta):
            raise NoiseFloorError(path, f"metric {metric_name!r} deltas must be numeric")
        parsed.append(float(delta))
    return tuple(parsed)


def _parse_seed_ids(path: Path, raw: Sequence[JsonScalar] | JsonScalar, seed_count: int) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise NoiseFloorError(path, "seed_ids must be a list")
    if len(raw) != seed_count:
        raise NoiseFloorError(path, "seed_ids length must equal seed_count")
    seed_ids: list[int] = []
    for seed_id in raw:
        if not isinstance(seed_id, int) or isinstance(seed_id, bool):
            raise NoiseFloorError(path, "seed_ids must contain integers")
        seed_ids.append(seed_id)
    return tuple(seed_ids)


def _parse_config_pairs(
    path: Path,
    raw: Sequence[Mapping[str, JsonScalar] | JsonScalar] | JsonScalar,
    seed_count: int,
) -> tuple[Mapping[str, str], ...]:
    if not isinstance(raw, list):
        raise NoiseFloorError(path, "identical_config_pairs must be a list")
    if len(raw) != seed_count:
        raise NoiseFloorError(path, "identical_config_pairs length must equal seed_count")
    pairs: list[Mapping[str, str]] = []
    for pair in raw:
        if not isinstance(pair, dict):
            raise NoiseFloorError(path, "each identical_config_pairs entry must be an object")
        left = pair.get("left")
        right = pair.get("right")
        if not isinstance(left, str) or not isinstance(right, str):
            raise NoiseFloorError(path, "config pairs require string left and right fields")
        if left == right:
            raise NoiseFloorError(path, "config pair paths must be distinct")
        pairs.append({"left": left, "right": right})
    return tuple(pairs)


def _parse_metric_map(
    path: Path,
    field_name: str,
    raw: Mapping[str, JsonScalar] | Sequence[JsonScalar] | JsonScalar,
) -> Mapping[str, float]:
    if not isinstance(raw, dict):
        raise NoiseFloorError(path, f"{field_name} must be a JSON object")
    parsed: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise NoiseFloorError(path, f"{field_name} metric names must be strings")
        if not _is_number(value):
            raise NoiseFloorError(path, f"{field_name} values must be numeric")
        parsed[name] = float(value)
    if not parsed:
        raise NoiseFloorError(path, f"{field_name} must not be empty")
    return parsed


def _is_number(value: JsonScalar) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit("CODEX: wire CLI args -> compare([Path(a) for a in argv[1:]])")
