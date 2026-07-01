from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pytest

from grip.eval.aggregate_headroom import (
    AggregateDecisionConfig,
    AggregateDecisionError,
    SeedDecision,
    aggregate_headroom_decision,
    write_aggregate_report,
)


SeedStatus = Literal["keep", "pivot", "blocked"]


def _seed_decision(seed: int, status: SeedStatus, interpretable: bool = True) -> SeedDecision:
    return SeedDecision(
        seed=seed,
        status=status,
        interpretable=interpretable,
        content_minus_dense=1.0 if status == "keep" else -1.0,
        authorize_avsb=status == "keep",
    )


def test_aggregate_headroom_authorizes_when_keep_rate_passes() -> None:
    # Given: eight interpretable seed decisions with six seed-level keeps.
    decisions = tuple(
        _seed_decision(seed, "keep" if seed < 6 else "pivot")
        for seed in range(8)
    )

    # When: the aggregate rule is evaluated.
    result = aggregate_headroom_decision(decisions)

    # Then: program-level A/B authorization is allowed.
    assert result.status == "keep"
    assert result.authorize_avsb is True
    assert result.keep_count == 6
    assert result.keep_rate == 0.75
    assert result.reason == "ok"


def test_aggregate_headroom_pivots_g013_reversal_shape() -> None:
    # Given: the G013 reversal shape with three seed-level keeps out of eight.
    decisions = tuple(
        _seed_decision(seed, "keep" if seed in {0, 1, 4} else "pivot")
        for seed in range(8)
    )

    # When: the aggregate rule is evaluated.
    result = aggregate_headroom_decision(decisions)

    # Then: seed-level keeps do not authorize program-level Grip A/B.
    assert result.status == "pivot"
    assert result.authorize_avsb is False
    assert result.keep_count == 3
    assert result.keep_rate == 0.375
    assert result.reason == "insufficient_keep_rate"


def test_aggregate_headroom_blocks_insufficient_seed_count() -> None:
    # Given: fewer than the preregistered minimum number of seeds.
    decisions = tuple(_seed_decision(seed, "keep") for seed in range(7))

    # When: the aggregate rule is evaluated.
    result = aggregate_headroom_decision(decisions)

    # Then: it blocks rather than extrapolating from too few seeds.
    assert result.status == "blocked"
    assert result.authorize_avsb is False
    assert result.reason == "insufficient_seed_count"


def test_aggregate_headroom_blocks_non_interpretable_seed() -> None:
    # Given: one seed-level result that did not pass the scorer gate.
    decisions = tuple(
        _seed_decision(seed, "keep", interpretable=seed != 7)
        for seed in range(8)
    )

    # When: the aggregate rule is evaluated.
    result = aggregate_headroom_decision(decisions)

    # Then: it blocks until every required seed is interpretable.
    assert result.status == "blocked"
    assert result.authorize_avsb is False
    assert result.reason == "insufficient_interpretable_rate"


def test_aggregate_headroom_blocks_when_not_preregistered() -> None:
    # Given: otherwise strong seed-level evidence but no aggregate preregistration.
    decisions = tuple(_seed_decision(seed, "keep") for seed in range(8))

    # When: the aggregate rule is evaluated without preregistration.
    result = aggregate_headroom_decision(
        decisions,
        AggregateDecisionConfig(preregistered=False),
    )

    # Then: it blocks program-level A/B authorization.
    assert result.status == "blocked"
    assert result.authorize_avsb is False
    assert result.reason == "aggregate_not_preregistered"


def test_aggregate_headroom_rejects_invalid_thresholds() -> None:
    # Given / When / Then: invalid aggregate thresholds fail at the boundary.
    with pytest.raises(AggregateDecisionError, match="minimum_keep_rate"):
        aggregate_headroom_decision((), AggregateDecisionConfig(minimum_keep_rate=1.1))
    with pytest.raises(AggregateDecisionError, match="minimum_interpretable_rate"):
        aggregate_headroom_decision(
            (),
            AggregateDecisionConfig(minimum_interpretable_rate=math.nan),
        )
    with pytest.raises(AggregateDecisionError, match="minimum_keep_rate"):
        aggregate_headroom_decision(
            (),
            AggregateDecisionConfig(minimum_keep_rate=math.inf),
        )


def test_write_aggregate_report(tmp_path: Path) -> None:
    # Given: an aggregate decision.
    result = aggregate_headroom_decision(
        tuple(_seed_decision(seed, "pivot") for seed in range(8))
    )

    # When: the report is written.
    path = write_aggregate_report(tmp_path / "aggregate.json", result)

    # Then: the artifact records authorization separately from seed status.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "pivot"
    assert payload["authorize_avsb"] is False
    assert payload["keep_count"] == 0
