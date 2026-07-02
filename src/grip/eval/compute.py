from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn

from .noise_floor import is_number
from .score_types import JsonValue, RunScore, ScoreArtifactError


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    parameter_count: int
    token_count: int
    estimated_forward_flops: int
    read_budget: int | None


def compute_budget(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    read_budget: int | None,
) -> ComputeBudget:
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    token_count = int(tokens.numel())
    return ComputeBudget(
        parameter_count=parameter_count,
        token_count=token_count,
        estimated_forward_flops=2 * parameter_count * token_count,
        read_budget=read_budget,
    )


def compute_payload(budget: ComputeBudget) -> Mapping[str, JsonValue]:
    return {
        "estimated_forward_flops": budget.estimated_forward_flops,
        "parameter_count": budget.parameter_count,
        "read_budget": budget.read_budget,
        "token_count": budget.token_count,
    }


def run_compute(run_dir: Path) -> Mapping[str, float | int | None]:
    eval_payload = _load_optional_json(run_dir / "eval_tensors.json")
    config_payload = _load_optional_json(run_dir / "config.resolved.json")
    raw_compute = eval_payload.get("compute") if isinstance(eval_payload, dict) else None
    compute = raw_compute if isinstance(raw_compute, dict) else {}
    token_count = _optional_number(compute.get("token_count"))
    if token_count is None and isinstance(eval_payload, dict):
        token_count = _optional_number(eval_payload.get("tokens"))
    read_budget = _optional_number(compute.get("read_budget"))
    if read_budget is None and isinstance(config_payload, dict):
        read_budget = _optional_number(config_payload.get("read_budget"))
    return {
        "estimated_forward_flops": _optional_number(compute.get("estimated_forward_flops")),
        "parameter_count": _optional_number(compute.get("parameter_count")),
        "read_budget": read_budget,
        "token_count": token_count,
    }


def compute_mismatches(scores: Sequence[RunScore], tolerance: float) -> tuple[str, ...]:
    mismatches: list[str] = []
    read_budgets = {score.compute.get("read_budget") for score in scores}
    if len(read_budgets) > 1:
        mismatches.append("compute.read_budget")
    for field in ("parameter_count", "estimated_forward_flops"):
        values = tuple((score.run_dir.name, score.compute.get(field)) for score in scores)
        missing = tuple(run_name for run_name, value in values if value is None)
        if missing:
            mismatches.extend(f"{run_name}.compute.{field}" for run_name in missing)
            continue
        parsed = tuple(float(value) for _, value in values if value is not None)
        if not parsed:
            continue
        maximum = max(parsed)
        minimum = min(parsed)
        if maximum == 0:
            if minimum != maximum:
                mismatches.append(f"compute.{field}")
            continue
        if (maximum - minimum) / maximum > tolerance:
            mismatches.append(f"compute.{field}")
    return tuple(sorted(set(mismatches)))


def _load_optional_json(path: Path) -> JsonValue:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoreArtifactError(path, "artifact must be valid JSON") from exc


def _optional_number(value: JsonValue) -> float | int | None:
    if value is None:
        return None
    if not is_number(value):
        return None
    return value
