from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Mapping, Sequence

from .experiment_config import experiment_provenance_payload
from .headroom_baselines import BASELINE_NAMES
from .score_types import JsonValue

if TYPE_CHECKING:
    from .noise_floor_calibration_types import NoiseFloorCalibrationConfig


NOISE_FLOOR_CONTENT_HASH_VERSION = "noise-floor-authority-v1"

_HASHED_AUTHORITY_FIELDS: tuple[str, ...] = (
    "calibration",
    "calibration_pairs",
    "metric_ceilings",
    "metric_deltas",
    "minimum_signal_threshold",
    "seed_count",
    "seed_ids",
    "zero_tolerance",
)


def calibration_payload(config: NoiseFloorCalibrationConfig) -> dict[str, JsonValue]:
    payload = experiment_provenance_payload(
        config,
        decision_seed_count=len(config.decision_seed_ids),
        include_device=True,
    )
    payload["baseline_names"] = list(BASELINE_NAMES)
    return payload


def noise_floor_payload(
    config: NoiseFloorCalibrationConfig,
    pairs: Sequence[dict[str, JsonValue]],
    metric_deltas: dict[str, tuple[float, ...]],
    metric_ceilings: dict[str, float],
    minimum_signal_threshold: dict[str, float],
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "calibration": calibration_payload(config),
        "calibration_pairs": list(pairs),
        "kind": "M-noise-floor",
        "metric_ceilings": metric_ceilings,
        "metric_deltas": {
            metric: list(deltas)
            for metric, deltas in metric_deltas.items()
        },
        "minimum_signal_threshold": minimum_signal_threshold,
        "seed_count": len(config.seed_ids),
        "seed_ids": list(config.seed_ids),
        "zero_tolerance": config.minimum_signal_floor,
    }
    return attach_noise_floor_content_hash(payload)


def attach_noise_floor_content_hash(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    payload["content_hash"] = noise_floor_content_hash(payload)
    payload["content_hash_version"] = NOISE_FLOOR_CONTENT_HASH_VERSION
    return payload


def noise_floor_content_hash(payload: Mapping[str, JsonValue]) -> str:
    authority = {
        field: payload[field]
        for field in _HASHED_AUTHORITY_FIELDS
        if field in payload
    }
    canonical = json.dumps(
        authority,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
