from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Mapping, Sequence

from .noise_floor_artifact import NOISE_FLOOR_CONTENT_HASH_VERSION, noise_floor_content_hash
from .score_types import JsonScalar, JsonValue, NoiseFloorArtifact, NoiseFloorError


MIN_NOISE_FLOOR_SEEDS: Final = 8


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
    _validate_content_hash(path, raw)
    seed_count = raw.get("seed_count")
    if not isinstance(seed_count, int) or isinstance(seed_count, bool):
        raise NoiseFloorError(path, "seed_count must be an integer")
    if seed_count < MIN_NOISE_FLOOR_SEEDS:
        raise NoiseFloorError(path, "seed_count must be >= 8")
    seed_ids = _parse_seed_ids(path, raw.get("seed_ids"), seed_count)
    calibration_pairs = _parse_calibration_pairs(path, raw.get("calibration_pairs"), seed_count)
    calibration = _parse_calibration(path, raw.get("calibration"))
    zero_tolerance = _parse_zero_tolerance(path, raw.get("zero_tolerance"))
    minimum_signal_threshold = _parse_metric_map(
        path,
        "minimum_signal_threshold",
        raw.get("minimum_signal_threshold"),
    )
    metric_ceilings = _parse_metric_map(path, "metric_ceilings", raw.get("metric_ceilings"))
    raw_deltas = raw.get("metric_deltas")
    if not isinstance(raw_deltas, dict):
        raise NoiseFloorError(path, "metric_deltas must be a JSON object")
    metric_deltas = _parse_metric_delta_map(
        path,
        raw_deltas,
        seed_count,
        metric_ceilings,
        minimum_signal_threshold,
        zero_tolerance,
    )
    return NoiseFloorArtifact(
        path=path,
        content_hash=raw["content_hash"],
        seed_count=seed_count,
        seed_ids=seed_ids,
        calibration_pairs=calibration_pairs,
        calibration=calibration,
        minimum_signal_threshold=minimum_signal_threshold,
        metric_deltas=metric_deltas,
        metric_ceilings=metric_ceilings,
        zero_tolerance=zero_tolerance,
    )


def is_number(value: JsonScalar) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _parse_metric_delta_map(
    path: Path,
    raw_deltas: Mapping[JsonScalar, JsonScalar],
    seed_count: int,
    metric_ceilings: Mapping[str, float],
    minimum_signal_threshold: Mapping[str, float],
    zero_tolerance: float,
) -> Mapping[str, tuple[float, ...]]:
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
        if expected_ceiling <= zero_tolerance:
            raise NoiseFloorError(path, f"metric {metric_name!r} has no measurable calibration spread")
        if metric_name not in minimum_signal_threshold:
            raise NoiseFloorError(path, f"metric {metric_name!r} missing minimum signal threshold")
        metric_deltas[metric_name] = parsed
    if not metric_deltas:
        raise NoiseFloorError(path, "at least one metric delta series is required")
    return metric_deltas


def _validate_content_hash(path: Path, raw: Mapping[str, JsonValue]) -> None:
    version = raw.get("content_hash_version")
    if version != NOISE_FLOOR_CONTENT_HASH_VERSION:
        raise NoiseFloorError(path, f"content_hash_version must be {NOISE_FLOOR_CONTENT_HASH_VERSION}")
    content_hash = raw.get("content_hash")
    if not isinstance(content_hash, str):
        raise NoiseFloorError(path, "content_hash must be a string")
    try:
        expected_hash = noise_floor_content_hash(raw)
    except (TypeError, ValueError) as exc:
        raise NoiseFloorError(path, "content_hash cannot be computed") from exc
    if content_hash != expected_hash:
        raise NoiseFloorError(path, "content_hash is stale")


def _parse_metric_deltas(
    path: Path,
    metric_name: str,
    deltas: Sequence[int | float],
) -> tuple[float, ...]:
    parsed: list[float] = []
    for delta in deltas:
        if not is_number(delta):
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


def _parse_calibration_pairs(
    path: Path,
    raw: Sequence[Mapping[str, JsonScalar] | JsonScalar] | JsonScalar,
    seed_count: int,
) -> tuple[Mapping[str, JsonScalar], ...]:
    if not isinstance(raw, list):
        raise NoiseFloorError(path, "calibration_pairs must be a list")
    if len(raw) != seed_count:
        raise NoiseFloorError(path, "calibration_pairs length must equal seed_count")
    pairs: list[Mapping[str, JsonScalar]] = []
    for pair in raw:
        if not isinstance(pair, dict):
            raise NoiseFloorError(path, "each calibration_pairs entry must be an object")
        left = pair.get("left")
        right = pair.get("right")
        left_seed = pair.get("left_seed")
        right_seed = pair.get("right_seed")
        if not isinstance(left, str) or not isinstance(right, str):
            raise NoiseFloorError(path, "config pairs require string left and right fields")
        if not isinstance(left_seed, int) or isinstance(left_seed, bool):
            raise NoiseFloorError(path, "calibration pairs require integer left_seed")
        if not isinstance(right_seed, int) or isinstance(right_seed, bool):
            raise NoiseFloorError(path, "calibration pairs require integer right_seed")
        if left == right:
            raise NoiseFloorError(path, "config pair paths must be distinct")
        if left_seed == right_seed:
            raise NoiseFloorError(path, "calibration pair seeds must be distinct")
        pairs.append({"left": left, "right": right, "left_seed": left_seed, "right_seed": right_seed})
    return tuple(pairs)


def _parse_zero_tolerance(path: Path, raw: JsonScalar) -> float:
    if not is_number(raw):
        raise NoiseFloorError(path, "zero_tolerance must be numeric")
    zero_tolerance = float(raw)
    if zero_tolerance < 0:
        raise NoiseFloorError(path, "zero_tolerance must be non-negative")
    return zero_tolerance


def _parse_calibration(path: Path, raw: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(raw, dict):
        raise NoiseFloorError(path, "calibration must be a JSON object")
    required = ("baseline_names", "data", "device", "eval", "model", "sparse", "train")
    missing = tuple(name for name in required if name not in raw)
    if missing:
        names = ", ".join(missing)
        raise NoiseFloorError(path, f"calibration missing required fields: {names}")
    baseline_names = raw.get("baseline_names")
    if not isinstance(baseline_names, list) or not all(isinstance(name, str) for name in baseline_names):
        raise NoiseFloorError(path, "calibration baseline_names must be a string list")
    return raw


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
        if not is_number(value):
            raise NoiseFloorError(path, f"{field_name} values must be numeric")
        parsed[name] = float(value)
    if not parsed:
        raise NoiseFloorError(path, f"{field_name} must not be empty")
    return parsed
