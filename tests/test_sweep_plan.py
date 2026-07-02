from __future__ import annotations

from dataclasses import replace
import json

import pytest

from grip.eval.sweep_plan import (
    SweepCriteria,
    SweepPlan,
    SweepPlanError,
    default_spec003_plan,
    validate_sweep_plan,
    write_sweep_plan,
)
from grip.eval.headroom_baselines import BASELINE_NAMES


SPEC003_CORE_VARIANTS = {
    *BASELINE_NAMES,
    "generic-memory",
    "grip-select-shuffle-grip",
    "grip-select-wrong-sample-grip",
    "grip-select-bottleneck-off",
}


def test_default_spec003_plan_contains_core_ablation_matrix():
    # Given: the default SPEC-003 sweep plan.
    plan = default_spec003_plan()

    # When: its variant names are collected.
    names = {variant.name for variant in plan.variants}

    # Then: every core backdoor-closing variant is present.
    assert names == SPEC003_CORE_VARIANTS
    assert plan.declaration_only is True
    assert plan.lead_task == "T1-source-reliability-reversal"
    assert "local" in names
    assert "local-only" not in names
    assert plan.read_k == (4, 8, 16)
    assert plan.grip_r == (2, 4)
    assert plan.seed_count >= 8


def test_validate_sweep_plan_rejects_missing_variant():
    # Given: a plan with one core variant removed.
    plan = default_spec003_plan()
    incomplete = SweepPlan(
        declaration_only=plan.declaration_only,
        lead_task=plan.lead_task,
        calibration_tasks=plan.calibration_tasks,
        sizes=plan.sizes,
        seq_lens=plan.seq_lens,
        read_k=plan.read_k,
        grip_r=plan.grip_r,
        seed_count=plan.seed_count,
        noise_floor_metric=plan.noise_floor_metric,
        min_signal_delta=plan.min_signal_delta,
        variants=plan.variants[:-1],
        criteria=plan.criteria,
    )

    # When / Then: validation rejects the missing backdoor intervention.
    with pytest.raises(SweepPlanError, match="missing variants"):
        validate_sweep_plan(incomplete)


def test_validate_sweep_plan_requires_matched_budget_for_comparison_variants():
    # Given: a plan whose A/B comparison has an unmatched read budget.
    plan = default_spec003_plan()
    broken_variants = tuple(
        replace(variant, read_budget=99)
        if variant.name == "grip-select-B"
        else variant
        for variant in plan.variants
    )
    broken = SweepPlan(
        declaration_only=plan.declaration_only,
        lead_task=plan.lead_task,
        calibration_tasks=plan.calibration_tasks,
        sizes=plan.sizes,
        seq_lens=plan.seq_lens,
        read_k=plan.read_k,
        grip_r=plan.grip_r,
        seed_count=plan.seed_count,
        noise_floor_metric=plan.noise_floor_metric,
        min_signal_delta=plan.min_signal_delta,
        variants=broken_variants,
        criteria=plan.criteria,
    )

    # When / Then: validation rejects the confounded comparison.
    with pytest.raises(SweepPlanError, match="matched read budget"):
        validate_sweep_plan(broken)


@pytest.mark.parametrize("variant_name", sorted(SPEC003_CORE_VARIANTS))
def test_validate_sweep_plan_requires_matched_flops_for_every_core_variant(variant_name):
    # Given: a core variant with unmatched FLOPs.
    plan = default_spec003_plan()
    broken_variants = tuple(
        replace(variant, matched_flops=False)
        if variant.name == variant_name
        else variant
        for variant in plan.variants
    )
    broken = replace(plan, variants=broken_variants)

    # When / Then: validation rejects the confounded row.
    with pytest.raises(SweepPlanError, match="matched FLOPs"):
        validate_sweep_plan(broken)


@pytest.mark.parametrize(
    "criteria",
    [
        SweepCriteria(True, True, False, True, "rule"),
        SweepCriteria(True, True, True, False, "rule"),
    ],
)
def test_validate_sweep_plan_requires_leakage_probe_gates(criteria):
    # Given: a plan with one required leakage gate disabled.
    plan = replace(default_spec003_plan(), criteria=criteria)

    # When / Then: validation rejects the missing leakage guard.
    with pytest.raises(SweepPlanError, match="leakage probes"):
        validate_sweep_plan(plan)


@pytest.mark.parametrize(
    "field_name,value,pattern",
    [
        ("sizes", ("1M",), "sizes"),
        ("seq_lens", (512,), "seq lengths"),
        ("calibration_tasks", (), "calibration"),
        ("noise_floor_metric", "", "noise floor"),
        ("min_signal_delta", -0.1, "signal delta"),
    ],
)
def test_validate_sweep_plan_rejects_incomplete_preregistration(field_name, value, pattern):
    # Given: a plan with a weakened preregistration field.
    plan = replace(default_spec003_plan(), **{field_name: value})

    # When / Then: validation rejects the incomplete grid.
    with pytest.raises(SweepPlanError, match=pattern):
        validate_sweep_plan(plan)


def test_validate_sweep_plan_rejects_duplicate_variants():
    # Given: a plan with duplicate variant names.
    plan = default_spec003_plan()
    duplicated = replace(plan, variants=plan.variants + (plan.variants[0],))

    # When / Then: validation rejects the duplicate matrix row.
    with pytest.raises(SweepPlanError, match="duplicate variants"):
        validate_sweep_plan(duplicated)


def test_validate_sweep_plan_requires_declaration_only_boundary():
    # Given: a SPEC-003 plan incorrectly marked as runner-consumable.
    plan = replace(default_spec003_plan(), declaration_only=False)

    # When / Then: validation preserves the declaration-only contract.
    with pytest.raises(SweepPlanError, match="declaration-only"):
        validate_sweep_plan(plan)


def test_sweep_plan_from_json_rejects_truthy_non_bool_declaration_only():
    # Given: a serialized plan with a truthy non-boolean declaration marker.
    plan = default_spec003_plan()
    payload = json.loads(plan.to_json())
    payload["declaration_only"] = "false"

    # When / Then: parsing rejects it before validation can treat it as true.
    with pytest.raises(SweepPlanError, match="JSON boolean"):
        SweepPlan.from_json(json.dumps(payload))


def test_write_sweep_plan_emits_preregistered_json(tmp_path):
    # Given: a valid SPEC-003 plan.
    plan = default_spec003_plan()

    # When: it is written as a preregistration artifact.
    path = write_sweep_plan(plan, tmp_path)

    # Then: the artifact can be reloaded and validated.
    loaded = SweepPlan.from_json(path.read_text())
    assert loaded == plan
    assert loaded.declaration_only is True
    assert validate_sweep_plan(loaded) is None
