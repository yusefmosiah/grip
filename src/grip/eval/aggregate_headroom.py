from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, Sequence

from .headroom_types import HeadroomStatus


AggregateStatus = Literal["keep", "pivot", "blocked"]


@dataclass(frozen=True, slots=True)
class AggregateDecisionError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class SeedDecision:
    seed: int
    status: HeadroomStatus
    interpretable: bool
    content_minus_dense: float
    authorize_avsb: bool


@dataclass(frozen=True, slots=True)
class AggregateDecisionConfig:
    minimum_seed_count: int = 8
    minimum_interpretable_rate: float = 1.0
    minimum_keep_rate: float = 0.75
    preregistered: bool = True


@dataclass(frozen=True, slots=True)
class AggregateDecision:
    status: AggregateStatus
    authorize_avsb: bool
    reason: str
    seed_count: int
    interpretable_count: int
    keep_count: int
    pivot_count: int
    blocked_count: int
    keep_rate: float
    interpretable_rate: float

    def to_json_text(self) -> str:
        return json.dumps(
            {
                "authorize_avsb": self.authorize_avsb,
                "blocked_count": self.blocked_count,
                "interpretable_count": self.interpretable_count,
                "interpretable_rate": self.interpretable_rate,
                "keep_count": self.keep_count,
                "keep_rate": self.keep_rate,
                "pivot_count": self.pivot_count,
                "reason": self.reason,
                "seed_count": self.seed_count,
                "status": self.status,
            },
            indent=2,
            sort_keys=True,
        )


def aggregate_headroom_decision(
    decisions: Sequence[SeedDecision],
    config: AggregateDecisionConfig = AggregateDecisionConfig(),
) -> AggregateDecision:
    _validate_config(config)
    if len(decisions) < config.minimum_seed_count:
        return _decision("blocked", "insufficient_seed_count", decisions)
    if not config.preregistered:
        return _decision("blocked", "aggregate_not_preregistered", decisions)
    interpretable_count = sum(1 for decision in decisions if decision.interpretable)
    interpretable_rate = interpretable_count / len(decisions)
    if interpretable_rate < config.minimum_interpretable_rate:
        return _decision("blocked", "insufficient_interpretable_rate", decisions)
    keep_count = sum(1 for decision in decisions if decision.status == "keep")
    keep_rate = keep_count / len(decisions)
    if keep_rate >= config.minimum_keep_rate:
        return _decision("keep", "ok", decisions)
    return _decision("pivot", "insufficient_keep_rate", decisions)


def write_aggregate_report(path: Path, decision: AggregateDecision) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(decision.to_json_text() + "\n", encoding="utf-8")
    return path


def _decision(
    status: AggregateStatus,
    reason: str,
    decisions: Sequence[SeedDecision],
) -> AggregateDecision:
    seed_count = len(decisions)
    interpretable_count = sum(1 for decision in decisions if decision.interpretable)
    keep_count = sum(1 for decision in decisions if decision.status == "keep")
    pivot_count = sum(1 for decision in decisions if decision.status == "pivot")
    blocked_count = sum(1 for decision in decisions if decision.status == "blocked")
    keep_rate = keep_count / seed_count if seed_count else 0.0
    interpretable_rate = interpretable_count / seed_count if seed_count else 0.0
    return AggregateDecision(
        status=status,
        authorize_avsb=status == "keep",
        reason=reason,
        seed_count=seed_count,
        interpretable_count=interpretable_count,
        keep_count=keep_count,
        pivot_count=pivot_count,
        blocked_count=blocked_count,
        keep_rate=keep_rate,
        interpretable_rate=interpretable_rate,
    )


def _validate_config(config: AggregateDecisionConfig) -> None:
    if config.minimum_seed_count <= 0:
        raise AggregateDecisionError("minimum_seed_count", "must be positive")
    _validate_rate("minimum_interpretable_rate", config.minimum_interpretable_rate)
    _validate_rate("minimum_keep_rate", config.minimum_keep_rate)


def _validate_rate(field: str, value: float) -> None:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise AggregateDecisionError(field, "must be between 0 and 1")
