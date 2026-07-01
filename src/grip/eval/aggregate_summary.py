from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, TypeAlias

from .aggregate_headroom import (
    AggregateDecision,
    AggregateDecisionConfig,
    SeedDecision,
    aggregate_headroom_decision,
    write_aggregate_report,
)
from .headroom_types import HeadroomStatus
from .noise_floor import is_number


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
SAFE_TASK_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class AggregateSummaryError(ValueError):
    path: Path
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class TaskAggregateReport:
    task: str
    decision: AggregateDecision
    report_path: Path


@dataclass(frozen=True, slots=True)
class AggregateSummaryResult:
    report_path: Path
    tasks: tuple[TaskAggregateReport, ...]


@dataclass(frozen=True, slots=True)
class _AggregateSummaryContext:
    out_dir: Path
    config: AggregateDecisionConfig


def aggregate_summary_file(
    summary_path: Path,
    out_dir: Path,
    config: AggregateDecisionConfig = AggregateDecisionConfig(),
) -> AggregateSummaryResult:
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AggregateSummaryError(summary_path, "summary", "must be valid JSON") from exc
    task_rows = _parse_summary(summary_path, raw)
    context = _AggregateSummaryContext(out_dir=out_dir, config=config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(
        _write_task_aggregate(task, decisions, context)
        for task, decisions in task_rows.items()
    )
    report_path = out_dir / "aggregate-summary.json"
    report_path.write_text(
        json.dumps(_summary_payload(tasks), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AggregateSummaryResult(report_path=report_path, tasks=tasks)


def _parse_summary(path: Path, raw: JsonValue) -> Mapping[str, tuple[SeedDecision, ...]]:
    if not isinstance(raw, dict):
        raise AggregateSummaryError(path, "summary", "must be a JSON object")
    if not raw:
        raise AggregateSummaryError(path, "summary", "must contain at least one task")
    parsed: dict[str, tuple[SeedDecision, ...]] = {}
    for task in sorted(raw):
        _parse_task_name(path, task)
        task_value = raw[task]
        if not isinstance(task_value, dict):
            raise AggregateSummaryError(path, task, "task entry must be a JSON object")
        rows = task_value.get("rows")
        if not isinstance(rows, list):
            raise AggregateSummaryError(path, f"{task}.rows", "must be a list")
        parsed[task] = tuple(
            _parse_row(path, f"{task}.rows[{index}]", row)
            for index, row in enumerate(rows)
        )
    return parsed


def _parse_row(path: Path, field: str, raw: JsonValue) -> SeedDecision:
    if not isinstance(raw, dict):
        raise AggregateSummaryError(path, field, "must be a JSON object")
    seed = _parse_seed(path, f"{field}.seed", raw.get("seed"))
    status = _parse_status(path, f"{field}.status", raw.get("status"))
    interpretable = _parse_bool(path, f"{field}.interpretable", raw.get("interpretable"))
    authorize_avsb = _parse_bool(path, f"{field}.authorize_avsb", raw.get("authorize_avsb"))
    if authorize_avsb is not (status == "keep"):
        raise AggregateSummaryError(
            path,
            f"{field}.authorize_avsb",
            "must match seed status",
        )
    content_minus_dense = _parse_float(
        path,
        f"{field}.content_minus_dense",
        raw.get("content_minus_dense"),
    )
    return SeedDecision(
        seed=seed,
        status=status,
        interpretable=interpretable,
        content_minus_dense=content_minus_dense,
        authorize_avsb=authorize_avsb,
    )


def _parse_task_name(path: Path, raw: str) -> str:
    if SAFE_TASK_PATTERN.fullmatch(raw) is None:
        raise AggregateSummaryError(path, "task", "must contain only letters, numbers, _, or -")
    return raw


def _parse_seed(path: Path, field: str, raw: JsonValue) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise AggregateSummaryError(path, field, "must be an integer")
    return raw


def _parse_status(path: Path, field: str, raw: JsonValue) -> HeadroomStatus:
    match raw:
        case "keep":
            return "keep"
        case "pivot":
            return "pivot"
        case "blocked":
            return "blocked"
        case _:
            raise AggregateSummaryError(path, field, "must be keep, pivot, or blocked")


def _parse_bool(path: Path, field: str, raw: JsonValue) -> bool:
    if not isinstance(raw, bool):
        raise AggregateSummaryError(path, field, "must be a boolean")
    return raw


def _parse_float(path: Path, field: str, raw: JsonValue) -> float:
    if not is_number(raw):
        raise AggregateSummaryError(path, field, "must be numeric")
    return float(raw)


def _write_task_aggregate(
    task: str,
    decisions: tuple[SeedDecision, ...],
    context: _AggregateSummaryContext,
) -> TaskAggregateReport:
    decision = aggregate_headroom_decision(decisions, context.config)
    report_path = write_aggregate_report(context.out_dir / f"{task}.aggregate.json", decision)
    return TaskAggregateReport(task=task, decision=decision, report_path=report_path)


def _summary_payload(tasks: tuple[TaskAggregateReport, ...]) -> dict[str, JsonValue]:
    return {
        "authorize_avsb": any(task.decision.authorize_avsb for task in tasks),
        "tasks": {
            task.task: _decision_payload(task.decision, task.report_path)
            for task in tasks
        },
    }


def _decision_payload(decision: AggregateDecision, report_path: Path) -> dict[str, JsonValue]:
    return {
        "authorize_avsb": decision.authorize_avsb,
        "blocked_count": decision.blocked_count,
        "interpretable_count": decision.interpretable_count,
        "interpretable_rate": decision.interpretable_rate,
        "keep_count": decision.keep_count,
        "keep_rate": decision.keep_rate,
        "pivot_count": decision.pivot_count,
        "reason": decision.reason,
        "report_path": str(report_path),
        "seed_count": decision.seed_count,
        "status": decision.status,
    }
