from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, TypeAlias, TypedDict


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


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
    calibration_pairs: tuple[Mapping[str, JsonScalar], ...]
    calibration: Mapping[str, JsonValue]
    minimum_signal_threshold: Mapping[str, float]
    metric_deltas: Mapping[str, tuple[float, ...]]
    metric_ceilings: Mapping[str, float]
    zero_tolerance: float


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    runs: tuple[RunScore, ...]
    interpretable: bool
    reason: str
    noise_floor: NoiseFloorArtifact | None
    config_mismatches: tuple[str, ...] = ()
    validity_failures: tuple[str, ...] = ()

    def to_json_text(self) -> str:
        payload = {
            "interpretable": self.interpretable,
            "config_mismatches": list(self.config_mismatches),
            "validity_failures": list(self.validity_failures),
            "noise_floor": _noise_floor_payload(self.noise_floor),
            "reason": self.reason,
            "runs": [
                {"metrics": dict(run.metrics), "run_dir": str(run.run_dir)}
                for run in self.runs
            ],
        }
        return json.dumps(payload, indent=2)


class NoiseFloorPayload(TypedDict):
    metric_ceilings: Mapping[str, float]
    minimum_signal_threshold: Mapping[str, float]
    path: str
    seed_count: int
    seed_ids: list[int]
    zero_tolerance: float


def _noise_floor_payload(noise_floor: NoiseFloorArtifact | None) -> NoiseFloorPayload | None:
    if noise_floor is None:
        return None
    return {
        "metric_ceilings": dict(noise_floor.metric_ceilings),
        "minimum_signal_threshold": dict(noise_floor.minimum_signal_threshold),
        "path": str(noise_floor.path),
        "seed_count": noise_floor.seed_count,
        "seed_ids": list(noise_floor.seed_ids),
        "zero_tolerance": noise_floor.zero_tolerance,
    }
