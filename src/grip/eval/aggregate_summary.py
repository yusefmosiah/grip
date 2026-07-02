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
    skipped_count: int
    skipped_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AggregateSummaryResult:
    report_path: Path
    tasks: tuple[TaskAggregateReport, ...]


@dataclass(frozen=True, slots=True)
class _AggregateSummaryContext:
    out_dir: Path
    config: AggregateDecisionConfig


@dataclass(frozen=True, slots=True)
class _ParsedTaskRows:
    decisions: tuple[SeedDecision, ...]
    skipped_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    decision: SeedDecision | None
    skipped_reason: str | None


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
        _write_task_aggregate(task, rows, context)
        for task, rows in task_rows.items()
    )
    report_path = out_dir / "aggregate-summary.json"
    report_path.write_text(
        json.dumps(_summary_payload(tasks), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AggregateSummaryResult(report_path=report_path, tasks=tasks)


def _parse_summary(path: Path, raw: JsonValue) -> Mapping[str, _ParsedTaskRows]:
    if not isinstance(raw, dict):
        raise AggregateSummaryError(path, "summary", "must be a JSON object")
    if not raw:
        raise AggregateSummaryError(path, "summary", "must contain at least one task")
    parsed: dict[str, _ParsedTaskRows] = {}
    for task in sorted(raw):
        _parse_task_name(path, task)
        task_value = raw[task]
        if not isinstance(task_value, dict):
            raise AggregateSummaryError(path, task, "task entry must be a JSON object")
        rows = task_value.get("rows")
        if not isinstance(rows, list):
            raise AggregateSummaryError(path, f"{task}.rows", "must be a list")
        parsed_rows = tuple(
            _parse_row(path, f"{task}.rows[{index}]", row)
            for index, row in enumerate(rows)
        )
        parsed[task] = _ParsedTaskRows(
            decisions=tuple(row.decision for row in parsed_rows if row.decision is not None),
            skipped_reasons=tuple(
                reason
                for row in parsed_rows
                if row.skipped_reason is not None
                for reason in (row.skipped_reason,)
            ),
        )
    return parsed


def _parse_row(path: Path, field: str, raw: JsonValue) -> _ParsedRow:
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
    if raw.get("unciteable") is True or raw.get("tier") == "smoke":
        return _ParsedRow(
            decision=None,
            skipped_reason=f"seed-{seed}:unciteable_smoke",
        )
    decision = SeedDecision(
        seed=seed,
        status=status,
        interpretable=interpretable,
        content_minus_dense=content_minus_dense,
        authorize_avsb=authorize_avsb,
    )
    return _ParsedRow(decision=decision, skipped_reason=None)


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
    rows: _ParsedTaskRows,
    context: _AggregateSummaryContext,
) -> TaskAggregateReport:
    decision = aggregate_headroom_decision(rows.decisions, context.config)
    report_path = write_aggregate_report(context.out_dir / f"{task}.aggregate.json", decision)
    return TaskAggregateReport(
        task=task,
        decision=decision,
        report_path=report_path,
        skipped_count=len(rows.skipped_reasons),
        skipped_reasons=rows.skipped_reasons,
    )


def _summary_payload(tasks: tuple[TaskAggregateReport, ...]) -> dict[str, JsonValue]:
    return {
        "authorize_avsb": any(task.decision.authorize_avsb for task in tasks),
        "tasks": {
            task.task: _decision_payload(task)
            for task in tasks
        },
    }


def _decision_payload(task: TaskAggregateReport) -> dict[str, JsonValue]:
    decision = task.decision
    return {
        "authorize_avsb": decision.authorize_avsb,
        "blocked_count": decision.blocked_count,
        "interpretable_count": decision.interpretable_count,
        "interpretable_rate": decision.interpretable_rate,
        "keep_count": decision.keep_count,
        "keep_rate": decision.keep_rate,
        "pivot_count": decision.pivot_count,
        "reason": decision.reason,
        "report_path": str(task.report_path),
        "seed_count": decision.seed_count,
        "skipped_count": task.skipped_count,
        "skipped_reasons": list(task.skipped_reasons),
        "status": decision.status,
    }
