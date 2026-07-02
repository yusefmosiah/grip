from __future__ import annotations

import json
from pathlib import Path

from grip.eval.score import compare, load_noise_floor, main, score_run

from score_fixtures import write_noise_floor as _write_noise_floor
from score_fixtures import write_run as _write_run


def test_score_run_loads_metrics_json(tmp_path: Path) -> None:
    # Given: a run directory with scorer-owned metrics.
    run_dir = _write_run(tmp_path / "run-a", {"accuracy": 0.75, "brier": 0.2})

    # When: the run is scored.
    result = score_run(run_dir)

    # Then: metrics are parsed from the artifact boundary.
    assert result.run_dir == run_dir
    assert result.metrics == {"accuracy": 0.75, "brier": 0.2}
    assert result.compute == {
        "estimated_forward_flops": 2_000,
        "parameter_count": 1_000,
        "read_budget": None,
        "token_count": 512,
    }


def test_compare_marks_missing_noise_floor_non_interpretable(tmp_path: Path) -> None:
    # Given: two scored runs but no M-noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})

    # When: the comparison is written.
    report = compare([run_a, run_b], tmp_path / "comparison.json")

    # Then: the comparison exists but cannot claim interpretable signal.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing"
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["interpretable"] is False
    assert saved["reason"] == "noise_floor_missing"


def test_compare_writes_only_explicit_output_path(tmp_path: Path) -> None:
    # Given: two scored runs and an explicit comparison path outside the run parent.
    run_a = _write_run(tmp_path / "runs" / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "runs" / "run-b", {"accuracy": 0.74})
    output_path = tmp_path / "reports" / "comparison.json"

    # When: the comparison is written.
    compare([run_a, run_b], output_path)

    # Then: the scorer writes only the requested artifact path.
    assert output_path.exists()
    assert not (tmp_path / "runs" / "comparison.json").exists()


def test_score_cli_writes_explicit_output_path(tmp_path: Path, capsys) -> None:
    # Given: two scored runs and a CLI output path.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    output_path = tmp_path / "cli" / "comparison.json"

    # When: the score CLI is invoked.
    exit_code = main([str(run_a), str(run_b), "--output", str(output_path)])

    # Then: it writes the requested comparison and prints its path.
    assert exit_code == 0
    assert Path(capsys.readouterr().out.strip()) == output_path
    assert json.loads(output_path.read_text(encoding="utf-8"))["reason"] == "noise_floor_missing"


def test_compare_keeps_valid_noise_floor_non_interpretable_without_preregistration(tmp_path: Path) -> None:
    # Given: two scored runs and a valid N>=8 M-noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is written with noise-floor evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the report records the noise-floor ceiling but cannot claim signal.
    assert report.interpretable is False
    assert report.reason == "comparison_not_preregistered"
    assert report.noise_floor is not None
    assert report.noise_floor.seed_count == 8
    assert report.noise_floor.metric_ceilings == {"accuracy": 0.02, "brier": 0.002}


def test_compare_allows_preregistered_interpretation_with_valid_noise_floor(tmp_path: Path) -> None:
    # Given: two scored runs and a valid N>=8 M-noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: the noise-floor gate authorizes interpretation.
    assert report.interpretable is True
    assert report.reason == "ok"
    assert report.compute_mismatches == ()


def test_compare_blocks_preregistered_noise_floor_without_run_metric_coverage(tmp_path: Path) -> None:
    # Given: two loss-scored runs and a schema-valid noise floor for a different metric.
    run_a = _write_run(tmp_path / "run-a", {"loss": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"loss": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: missing metric coverage still blocks interpretation.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing_metric"


def test_compare_blocks_mismatched_noise_floor_config(tmp_path: Path) -> None:
    # Given: two scored runs and a noise floor calibrated for a different sequence length.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["calibration"]["data"]["seq_len"] = 1024
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: the scorer blocks interpretation and names the mismatched fields.
    assert report.interpretable is False
    assert report.reason == "noise_floor_config_mismatch"
    assert report.config_mismatches == ("run-a.data.seq_len", "run-b.data.seq_len")


def test_compare_blocks_smoke_tier_runs(tmp_path: Path) -> None:
    # Given: scored runs marked as smoke-tier artifacts.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70}, valid=False)
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74}, valid=False)
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["calibration"]["data"]["seq_len"] = 8
    payload["calibration"]["decision"]["seed_count"] = 1
    payload["calibration"]["eval"]["batch_size"] = 1
    payload["calibration"]["train"]["batch_size"] = 1
    payload["calibration"]["train"]["steps"] = 0
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: smoke-tier artifacts cannot authorize interpretation.
    assert report.interpretable is False
    assert report.reason == "below_minimum_validity"
    assert report.validity_failures == (
        "run-a.data.seq_len",
        "run-a.decision.seed_count",
        "run-a.eval.batch_size",
        "run-a.train.batch_size",
        "run-a.train.steps",
        "run-b.data.seq_len",
        "run-b.decision.seed_count",
        "run-b.eval.batch_size",
        "run-b.train.batch_size",
        "run-b.train.steps",
    )


def test_compare_does_not_require_noise_floor_for_token_bookkeeping(tmp_path: Path) -> None:
    # Given: two loss-scored runs that also include token bookkeeping.
    run_a = _write_run(tmp_path / "run-a", {"loss": 0.70, "tokens": 8.0})
    run_b = _write_run(tmp_path / "run-b", {"loss": 0.74, "tokens": 8.0})
    noise_floor_path = tmp_path / "noise-floor.json"
    payload = {
        "kind": "M-noise-floor",
        "calibration": {
            "baseline_names": ["run-a", "run-b"],
            "decision": {"seed_count": 8},
            "data": {"seq_len": 512, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": 8, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": 8, "lr": 1e-3, "steps": 1000},
        },
        "seed_count": 8,
        "seed_ids": list(range(8)),
        "calibration_pairs": [
            {
                "left": f"seed-{seed}-a",
                "left_seed": seed + 100,
                "right": f"seed-{seed}-b",
                "right_seed": seed + 200,
            }
            for seed in range(8)
        ],
        "minimum_signal_threshold": {"loss": 0.02},
        "metric_ceilings": {"loss": 0.02},
        "metric_deltas": {"loss": [0.01, -0.02, 0.0, 0.015, -0.01, 0.005, 0.02, -0.015]},
        "zero_tolerance": 1e-12,
    }
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: token bookkeeping does not block loss-covered interpretation.
    assert report.interpretable is True
    assert report.reason == "ok"


def test_compare_blocks_threshold_only_noise_floor_metric(tmp_path: Path) -> None:
    # Given: loss-scored runs and a noise floor with loss threshold but no loss deltas.
    run_a = _write_run(tmp_path / "run-a", {"loss": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"loss": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["minimum_signal_threshold"]["loss"] = 0.02
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path, preregistered=True)

    # Then: threshold-only coverage does not authorize interpretation.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing_metric"


def test_compare_rejects_below_floor_noise_artifact_without_raising(tmp_path: Path) -> None:
    # Given: two scored runs and an invalid under-seeded noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json", seed_count=7)

    # When: the comparison is written with invalid noise-floor evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the comparison is explicitly non-interpretable.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["interpretable"] is False
    assert saved["reason"] == "noise_floor_invalid"


def test_compare_rejects_short_metric_delta_series_without_raising(tmp_path: Path) -> None:
    # Given: a noise-floor artifact with fewer metric deltas than seeds.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["metric_deltas"]["accuracy"] = [0.0] * 7
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is written with stale noise-floor evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the comparison is explicitly non-interpretable.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"


def test_compare_rejects_malformed_noise_artifact_without_raising(tmp_path: Path) -> None:
    # Given: two scored runs and a malformed noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = tmp_path / "noise-floor.json"
    noise_floor_path.write_text("{not json", encoding="utf-8")

    # When: the comparison is written with malformed noise-floor evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the comparison is explicitly non-interpretable.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["interpretable"] is False
    assert saved["reason"] == "noise_floor_invalid"


def test_load_noise_floor_requires_complete_authority_schema(tmp_path: Path) -> None:
    # Given: a noise-floor artifact without seed ids and config-pair authority.
    noise_floor_path = tmp_path / "noise-floor.json"
    noise_floor_path.write_text(
        json.dumps(
            {
                "kind": "M-noise-floor",
                "seed_count": 8,
                "metric_deltas": {"accuracy": [0.0] * 8},
            }
        ),
        encoding="utf-8",
    )

    # When: the comparison is written with incomplete noise-floor evidence.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the incomplete schema cannot authorize interpretation.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"


def test_compare_rejects_stale_metric_ceiling_without_raising(tmp_path: Path) -> None:
    # Given: a noise-floor artifact whose stored ceiling no longer matches deltas.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["metric_ceilings"]["accuracy"] = 0.001
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is written with stale noise-floor evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the comparison is explicitly non-interpretable.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"


def test_load_noise_floor_rejects_boolean_metric_values(tmp_path: Path) -> None:
    # Given: a noise-floor artifact with boolean metric values.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["minimum_signal_threshold"]["accuracy"] = True
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})

    # When: the comparison is written with invalid numeric evidence.
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)

    # Then: the comparison is explicitly non-interpretable.
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"


def test_load_noise_floor_requires_metric_deltas_for_each_seed(tmp_path: Path) -> None:
    # Given: a valid noise-floor artifact with metric deltas.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the artifact is loaded.
    artifact = load_noise_floor(noise_floor_path)

    # Then: ceilings are derived from absolute identical-config deltas.
    assert artifact.seed_count == 8
    assert artifact.seed_ids == tuple(range(8))
    assert len(artifact.calibration_pairs) == 8
    assert artifact.minimum_signal_threshold == {"accuracy": 0.02, "brier": 0.002}
    assert artifact.metric_ceilings["accuracy"] == 0.02


def test_load_noise_floor_rejects_degenerate_metric_spread(tmp_path: Path) -> None:
    # Given: a syntactically complete noise floor whose metric deltas are all numeric zero.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["metric_deltas"]["accuracy"] = [0.0] * 8
    payload["metric_ceilings"]["accuracy"] = 0.0
    payload["minimum_signal_threshold"]["accuracy"] = 0.0
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When / Then: loading rejects the degenerate calibration.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    report = compare([run_a, run_b], tmp_path / "comparison.json", noise_floor_path=noise_floor_path)
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"
