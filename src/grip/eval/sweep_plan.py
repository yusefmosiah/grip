from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final


CORE_VARIANTS: Final = (
    "dense",
    "local-only",
    "content-sparse",
    "generic-memory",
    "grip-read-A",
    "grip-select-B",
    "grip-select-shuffle-grip",
    "grip-select-wrong-sample-grip",
    "grip-select-bottleneck-off",
)


@dataclass(frozen=True, slots=True)
class SweepVariant:
    name: str
    selection: str
    grip_read: str
    aux_supervision: bool
    read_budget: int
    matched_flops: bool
    purpose: str


@dataclass(frozen=True, slots=True)
class SweepCriteria:
    signal_delta_must_exceed_noise_floor: bool
    require_task_specificity: bool
    require_grip_answer_probe_noise_floor: bool
    require_bypass_probe_noise_floor: bool
    decision_rule: str


@dataclass(frozen=True, slots=True)
class SweepPlan:
    lead_task: str
    calibration_tasks: tuple[str, ...]
    sizes: tuple[str, ...]
    seq_lens: tuple[int, ...]
    read_k: tuple[int, ...]
    grip_r: tuple[int, ...]
    seed_count: int
    noise_floor_metric: str
    min_signal_delta: float
    variants: tuple[SweepVariant, ...]
    criteria: SweepCriteria

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SweepPlan:
        payload = json.loads(raw)
        variants = tuple(SweepVariant(**variant) for variant in payload["variants"])
        criteria = SweepCriteria(**payload["criteria"])
        return SweepPlan(
            lead_task=payload["lead_task"],
            calibration_tasks=tuple(payload["calibration_tasks"]),
            sizes=tuple(payload["sizes"]),
            seq_lens=tuple(payload["seq_lens"]),
            read_k=tuple(payload["read_k"]),
            grip_r=tuple(payload["grip_r"]),
            seed_count=payload["seed_count"],
            noise_floor_metric=payload["noise_floor_metric"],
            min_signal_delta=payload["min_signal_delta"],
            variants=variants,
            criteria=criteria,
        )


class SweepPlanError(ValueError):
    pass


def default_spec003_plan() -> SweepPlan:
    read_budget = 16
    variants = (
        SweepVariant("dense", "full", "none", False, read_budget, True, "upper reference"),
        SweepVariant("local-only", "window", "none", False, read_budget, True, "cheap floor"),
        SweepVariant("content-sparse", "top-K content", "none", False, read_budget, True, "stock baseline"),
        SweepVariant("generic-memory", "top-K content", "generic slots", False, read_budget, True, "capacity confound"),
        SweepVariant("grip-read-A", "top-K content", "grip", True, read_budget, True, "aux supervision confound"),
        SweepVariant("grip-select-B", "top-K content + lambda grip", "grip", True, read_budget, True, "mechanism"),
        SweepVariant("grip-select-shuffle-grip", "B with shuffled grip", "grip", True, read_budget, True, "time-shuffle causal check"),
        SweepVariant("grip-select-wrong-sample-grip", "B with wrong sample grip", "grip", True, read_budget, True, "sample-causal check"),
        SweepVariant("grip-select-bottleneck-off", "B unconstrained grip", "grip", True, read_budget, True, "content-leakage check"),
    )
    return SweepPlan(
        lead_task="T1-source-reliability-reversal",
        calibration_tasks=("T0-bayesian-evidence-streams",),
        sizes=("1M", "4M", "16M"),
        seq_lens=(512, 1024),
        read_k=(4, 8, 16),
        grip_r=(2, 4),
        seed_count=8,
        noise_floor_metric="M-noise-floor",
        min_signal_delta=0.0,
        variants=variants,
        criteria=SweepCriteria(
            signal_delta_must_exceed_noise_floor=True,
            require_task_specificity=True,
            require_grip_answer_probe_noise_floor=True,
            require_bypass_probe_noise_floor=True,
            decision_rule="keep only if B beats A on T1 above noise floor across the K,R curve",
        ),
    )


def validate_sweep_plan(plan: SweepPlan) -> None:
    variant_names = [variant.name for variant in plan.variants]
    names = set(variant_names)
    if len(variant_names) != len(names):
        msg = "duplicate variants are not allowed"
        raise SweepPlanError(msg)
    missing = sorted(set(CORE_VARIANTS) - names)
    if missing:
        msg = f"missing variants: {', '.join(missing)}"
        raise SweepPlanError(msg)
    if plan.lead_task != "T1-source-reliability-reversal":
        msg = "SPEC-003 lead task must be T1-source-reliability-reversal"
        raise SweepPlanError(msg)
    if not plan.calibration_tasks:
        msg = "calibration tasks must be preregistered"
        raise SweepPlanError(msg)
    if plan.sizes != ("1M", "4M", "16M"):
        msg = "sizes must include the preregistered 1M, 4M, 16M grid"
        raise SweepPlanError(msg)
    if plan.seq_lens != (512, 1024):
        msg = "seq lengths must include the preregistered 512 and 1024 grid"
        raise SweepPlanError(msg)
    if plan.seed_count < 8:
        msg = "seed count must be preregistered at >=8"
        raise SweepPlanError(msg)
    if not plan.noise_floor_metric:
        msg = "noise floor metric must be preregistered"
        raise SweepPlanError(msg)
    if plan.min_signal_delta < 0:
        msg = "signal delta must be non-negative"
        raise SweepPlanError(msg)
    if plan.read_k != (4, 8, 16) or plan.grip_r != (2, 4):
        msg = "plan must report the full K,R curve"
        raise SweepPlanError(msg)
    _validate_matched_budget(plan)
    if not plan.criteria.signal_delta_must_exceed_noise_floor:
        msg = "signal deltas must exceed the noise floor"
        raise SweepPlanError(msg)
    if not plan.criteria.require_task_specificity:
        msg = "task specificity criterion is required"
        raise SweepPlanError(msg)
    if not plan.criteria.decision_rule:
        msg = "decision rule must be preregistered"
        raise SweepPlanError(msg)
    if (
        not plan.criteria.require_grip_answer_probe_noise_floor
        or not plan.criteria.require_bypass_probe_noise_floor
    ):
        msg = "leakage probes must be required"
        raise SweepPlanError(msg)


def _validate_matched_budget(plan: SweepPlan) -> None:
    variants = {variant.name: variant for variant in plan.variants}
    if not all(variant.matched_flops for variant in variants.values()):
        msg = "all core variants must be matched FLOPs"
        raise SweepPlanError(msg)
    grip_read = variants["grip-read-A"]
    grip_select = variants["grip-select-B"]
    content_sparse = variants["content-sparse"]
    budgets = {grip_read.read_budget, grip_select.read_budget, content_sparse.read_budget}
    if len(budgets) != 1:
        msg = "A, B, and content-sparse must have matched read budget"
        raise SweepPlanError(msg)


def write_sweep_plan(plan: SweepPlan, out_dir: Path) -> Path:
    validate_sweep_plan(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spec-003-sweep-plan.json"
    path.write_text(plan.to_json())
    return path
