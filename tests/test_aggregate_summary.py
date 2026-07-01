from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.aggregate_summary import (
    AggregateSummaryError,
    aggregate_summary_file,
)


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _row(seed: int, status: str, delta: float, interpretable: bool = True) -> dict[str, JsonValue]:
    return {
        "authorize_avsb": status == "keep",
        "content_minus_dense": delta,
        "interpretable": interpretable,
        "seed": seed,
        "status": status,
    }


def test_aggregate_summary_file_writes_task_reports_for_g013_shape(tmp_path: Path) -> None:
    # Given: a G013-style calibrated summary with Bayesian 0/8 keep and reversal 3/8 keep.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "bayesian": {
                    "rows": tuple(_row(seed, "pivot", -1.0) for seed in range(8)),
                },
                "reversal": {
                    "rows": tuple(
                        _row(seed, "keep" if seed in {0, 1, 4} else "pivot", 1.0)
                        for seed in range(8)
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    # When: aggregate reports are generated from the summary file.
    result = aggregate_summary_file(summary_path, tmp_path / "aggregate")

    # Then: both tasks remain unauthorized at the program level.
    assert result.report_path == tmp_path / "aggregate" / "aggregate-summary.json"
    assert tuple(task.task for task in result.tasks) == ("bayesian", "reversal")
    assert all(task.decision.status == "pivot" for task in result.tasks)
    assert all(task.decision.authorize_avsb is False for task in result.tasks)
    assert (tmp_path / "aggregate" / "bayesian.aggregate.json").exists()
    assert (tmp_path / "aggregate" / "reversal.aggregate.json").exists()
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["authorize_avsb"] is False
    assert payload["tasks"]["reversal"]["keep_count"] == 3


def test_aggregate_summary_file_blocks_non_interpretable_rows(tmp_path: Path) -> None:
    # Given: a summary with one non-interpretable seed row.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "reversal": {
                    "rows": tuple(
                        _row(seed, "keep", 1.0, interpretable=seed != 7)
                        for seed in range(8)
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    # When: aggregate reports are generated.
    result = aggregate_summary_file(summary_path, tmp_path / "aggregate")

    # Then: the task blocks instead of authorizing from partial evidence.
    assert result.tasks[0].decision.status == "blocked"
    assert result.tasks[0].decision.authorize_avsb is False
    assert result.tasks[0].decision.reason == "insufficient_interpretable_rate"


def test_aggregate_summary_file_rejects_missing_rows(tmp_path: Path) -> None:
    # Given: a summary task entry without rows.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"bayesian": {}}), encoding="utf-8")

    # When / Then: the boundary rejects the malformed artifact.
    with pytest.raises(AggregateSummaryError, match="rows"):
        aggregate_summary_file(summary_path, tmp_path / "aggregate")


def test_aggregate_summary_file_rejects_empty_summary(tmp_path: Path) -> None:
    # Given: an empty summary with no task decisions.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({}), encoding="utf-8")

    # When / Then: the boundary rejects it instead of writing an empty authorization artifact.
    with pytest.raises(AggregateSummaryError, match="summary"):
        aggregate_summary_file(summary_path, tmp_path / "aggregate")


def test_aggregate_summary_file_rejects_invalid_json(tmp_path: Path) -> None:
    # Given: a corrupt summary artifact.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{not json", encoding="utf-8")

    # When / Then: the boundary raises a typed aggregate summary error.
    with pytest.raises(AggregateSummaryError, match="valid JSON"):
        aggregate_summary_file(summary_path, tmp_path / "aggregate")


def test_aggregate_summary_file_rejects_unsafe_task_key(tmp_path: Path) -> None:
    # Given: a summary whose task key would escape the aggregate output directory.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"../escaped": {"rows": tuple(_row(seed, "pivot", -1.0) for seed in range(8))}}),
        encoding="utf-8",
    )

    # When / Then: the boundary rejects the key before writing reports.
    with pytest.raises(AggregateSummaryError, match="task"):
        aggregate_summary_file(summary_path, tmp_path / "aggregate")
    assert not (tmp_path / "escaped.aggregate.json").exists()


def test_aggregate_summary_file_rejects_status_authorization_mismatch(tmp_path: Path) -> None:
    # Given: a row whose seed-level status and authorization fields disagree.
    summary_path = tmp_path / "summary.json"
    rows = [_row(seed, "pivot", -1.0) for seed in range(8)]
    rows[0]["status"] = "keep"
    rows[0]["authorize_avsb"] = False
    summary_path.write_text(json.dumps({"reversal": {"rows": rows}}), encoding="utf-8")

    # When / Then: the boundary rejects the contradictory seed row.
    with pytest.raises(AggregateSummaryError, match="authorize_avsb"):
        aggregate_summary_file(summary_path, tmp_path / "aggregate")
