from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from grip.eval.score import compare, load_noise_floor, score_run


def _write_run(run_dir: Path, metrics: Mapping[str, float]) -> Path:
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
    (run_dir / "config.resolved.json").write_text(
        json.dumps(_run_config(run_dir.name), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _run_config(name: str, *, seq_len: int = 8) -> dict:
    return {
        "data": {"seq_len": seq_len, "task": "bayesian", "vocab_size": 17},
        "eval": {"batch_size": 1, "seed": 10_000, "seed_offset": 10_000},
        "model": {
            "attention_mode": None,
            "d_model": 16,
            "n_heads": 4,
            "n_hypotheses": 3,
            "n_layers": 1,
            "name": name,
        },
        "read_budget": None,
        "run": {"device": "cpu", "mode": "preregistered"},
        "seed": 0,
        "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
        "train": {"batch_size": 1, "lr": 1e-3, "steps": 0},
    }


def _write_noise_floor(path: Path, *, seed_count: int = 8) -> Path:
    accuracy_deltas = [0.01, -0.02, 0.0, 0.015, -0.01, 0.005, 0.02, -0.015][:seed_count]
    brier_deltas = [0.001, -0.002, 0.0, 0.0015, -0.001, 0.0005, 0.002, -0.0015][:seed_count]
    payload = {
        "kind": "M-noise-floor",
        "calibration": {
            "baseline_names": ["run-a", "run-b", "dense", "local", "content-sparse"],
            "data": {"seq_len": 8, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": 1, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": 1, "lr": 1e-3, "steps": 0},
        },
        "seed_count": seed_count,
        "seed_ids": list(range(seed_count)),
        "calibration_pairs": [
            {
                "left": f"seed-{seed}-a",
                "left_seed": seed + 100,
                "right": f"seed-{seed}-b",
                "right_seed": seed + 200,
            }
            for seed in range(seed_count)
        ],
        "minimum_signal_threshold": {"accuracy": 0.02, "brier": 0.002},
        "metric_ceilings": {"accuracy": 0.02, "brier": 0.002},
        "metric_deltas": {
            "accuracy": accuracy_deltas,
            "brier": brier_deltas,
        },
        "zero_tolerance": 1e-12,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_score_run_loads_metrics_json(tmp_path: Path) -> None:
    # Given: a run directory with scorer-owned metrics.
    run_dir = _write_run(tmp_path / "run-a", {"accuracy": 0.75, "brier": 0.2})

    # When: the run is scored.
    result = score_run(run_dir)

    # Then: metrics are parsed from the artifact boundary.
    assert result.run_dir == run_dir
    assert result.metrics == {"accuracy": 0.75, "brier": 0.2}


def test_compare_marks_missing_noise_floor_non_interpretable(tmp_path: Path) -> None:
    # Given: two scored runs but no M-noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})

    # When: the comparison is written.
    report = compare([run_a, run_b])

    # Then: the comparison exists but cannot claim interpretable signal.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing"
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["interpretable"] is False
    assert saved["reason"] == "noise_floor_missing"


def test_compare_keeps_valid_noise_floor_non_interpretable_without_preregistration(tmp_path: Path) -> None:
    # Given: two scored runs and a valid N>=8 M-noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is written with noise-floor evidence.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: the noise-floor gate authorizes interpretation.
    assert report.interpretable is True
    assert report.reason == "ok"


def test_compare_blocks_preregistered_noise_floor_without_run_metric_coverage(tmp_path: Path) -> None:
    # Given: two loss-scored runs and a schema-valid noise floor for a different metric.
    run_a = _write_run(tmp_path / "run-a", {"loss": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"loss": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: missing metric coverage still blocks interpretation.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing_metric"


def test_compare_blocks_mismatched_noise_floor_config(tmp_path: Path) -> None:
    # Given: two scored runs and a noise floor calibrated for a different sequence length.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    payload = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    payload["calibration"]["data"]["seq_len"] = 16
    noise_floor_path.write_text(json.dumps(payload), encoding="utf-8")

    # When: the comparison is explicitly marked preregistered.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: the scorer blocks interpretation and names the mismatched fields.
    assert report.interpretable is False
    assert report.reason == "noise_floor_config_mismatch"
    assert report.config_mismatches == ("run-a.data.seq_len", "run-b.data.seq_len")


def test_compare_does_not_require_noise_floor_for_token_bookkeeping(tmp_path: Path) -> None:
    # Given: two loss-scored runs that also include token bookkeeping.
    run_a = _write_run(tmp_path / "run-a", {"loss": 0.70, "tokens": 8.0})
    run_b = _write_run(tmp_path / "run-b", {"loss": 0.74, "tokens": 8.0})
    noise_floor_path = tmp_path / "noise-floor.json"
    payload = {
        "kind": "M-noise-floor",
        "calibration": {
            "baseline_names": ["run-a", "run-b"],
            "data": {"seq_len": 8, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": 1, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": 1, "lr": 1e-3, "steps": 0},
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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path, preregistered=True)

    # Then: threshold-only coverage does not authorize interpretation.
    assert report.interpretable is False
    assert report.reason == "noise_floor_missing_metric"


def test_compare_rejects_below_floor_noise_artifact_without_raising(tmp_path: Path) -> None:
    # Given: two scored runs and an invalid under-seeded noise-floor artifact.
    run_a = _write_run(tmp_path / "run-a", {"accuracy": 0.70})
    run_b = _write_run(tmp_path / "run-b", {"accuracy": 0.74})
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json", seed_count=7)

    # When: the comparison is written with invalid noise-floor evidence.
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)

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
    report = compare([run_a, run_b], noise_floor_path=noise_floor_path)
    assert report.interpretable is False
    assert report.reason == "noise_floor_invalid"
