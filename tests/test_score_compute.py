from __future__ import annotations

import json
from pathlib import Path

from grip.eval.headroom import MRegimeConfig, run_m_regime_smoke
from grip.eval.noise_floor_artifact import BASELINE_NAMES
from grip.eval.score import compare

from score_fixtures import write_noise_floor
from score_fixtures import write_run


def test_compare_records_compute_in_comparison_report(tmp_path: Path) -> None:
    # Given: two matched-compute runs and a valid noise-floor artifact.
    run_a = write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is written.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: each run carries compute accounting into the scorer-owned report.
    assert report.interpretable is True
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["compute_tolerance"] == 0.05
    assert saved["compute_mismatches"] == []
    assert saved["runs"][0]["compute"]["parameter_count"] == 1_000
    assert saved["runs"][1]["compute"]["estimated_forward_flops"] == 2_000


def test_compare_blocks_compute_mismatch_above_tolerance(tmp_path: Path) -> None:
    # Given: valid runs whose parameter and forward-FLOP estimates are unmatched.
    run_a = write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = write_run(
        tmp_path / "run-b",
        {"accuracy": 0.74},
        parameter_count=1_200,
        estimated_forward_flops=2_400,
    )
    noise_floor_path = write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: compute mismatch blocks interpretation before any winner claim.
    assert report.interpretable is False
    assert report.reason == "compute_mismatch"
    assert report.compute_mismatches == ("compute.estimated_forward_flops", "compute.parameter_count")


def test_compare_blocks_read_budget_mismatch(tmp_path: Path) -> None:
    # Given: valid runs with matched parameters/FLOPs but mismatched read budgets.
    run_a = write_run(tmp_path / "run-a", {"accuracy": 0.70}, read_budget=1)
    run_b = write_run(tmp_path / "run-b", {"accuracy": 0.74}, read_budget=8)
    noise_floor_path = write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: read-budget mismatch blocks interpretation as a compute mismatch.
    assert report.interpretable is False
    assert report.reason == "compute_mismatch"
    assert report.compute_mismatches == ("compute.read_budget",)


def test_headroom_writer_emits_compute_accounting(tmp_path: Path) -> None:
    # Given: the real headroom artifact writer on a tiny smoke config.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", top_k_blocks=2)

    # When: the M-regime path writes baseline artifacts and comparison JSON.
    result = run_m_regime_smoke(config)

    # Then: every declared baseline records compute in eval and comparison artifacts.
    assert {run_dir.name for run_dir in result.run_dirs} == set(BASELINE_NAMES)
    comparison = json.loads((config.out_dir / "comparison.json").read_text(encoding="utf-8"))
    comparison_by_name = {
        Path(record["run_dir"]).name: record["compute"]
        for record in comparison["runs"]
    }
    for run_dir in result.run_dirs:
        eval_payload = json.loads((run_dir / "eval_tensors.json").read_text(encoding="utf-8"))
        compute = eval_payload["compute"]
        expected_read_budget = None if run_dir.name == "dense" else 2
        assert compute["parameter_count"] > 0
        assert compute["estimated_forward_flops"] > 0
        assert compute["token_count"] == eval_payload["tokens"]
        assert compute["read_budget"] == expected_read_budget
        assert comparison_by_name[run_dir.name] == compute
