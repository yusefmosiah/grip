from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import grip.eval.headroom_runs as headroom_runs
from grip.eval.headroom import MRegimeConfig, run_m_regime_smoke
from grip.eval.headroom_types import BaselineSpec
from grip.eval.noise_floor_artifact import BASELINE_NAMES
from grip.models import ContentSparseTransformer, DenseTransformer


BaselineModel = DenseTransformer | ContentSparseTransformer


def _write_noise_floor(
    path: Path,
    *,
    train_steps: int = 0,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
) -> Path:
    payload = {
        "calibration": {
            "baseline_names": list(BASELINE_NAMES),
            "decision": {"seed_count": 1},
            "data": {"seq_len": 8, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": eval_batch_size, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": train_batch_size, "lr": 1e-3, "steps": train_steps},
        },
        "calibration_pairs": [
            {
                "left": f"seed-{seed}-a",
                "left_seed": seed + 100,
                "right": f"seed-{seed}-b",
                "right_seed": seed + 200,
            }
            for seed in range(8)
        ],
        "kind": "M-noise-floor",
        "metric_ceilings": {"loss": 0.01},
        "metric_deltas": {"loss": [0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.001]},
        "minimum_signal_threshold": {"loss": 0.01},
        "seed_count": 8,
        "seed_ids": list(range(8)),
        "zero_tolerance": 1e-12,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_accuracy_only_noise_floor(path: Path) -> Path:
    payload = {
        "calibration": {
            "baseline_names": list(BASELINE_NAMES),
            "decision": {"seed_count": 1},
            "data": {"seq_len": 8, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": 1, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": 1, "lr": 1e-3, "steps": 0},
        },
        "calibration_pairs": [
            {
                "left": f"seed-{seed}-a",
                "left_seed": seed + 100,
                "right": f"seed-{seed}-b",
                "right_seed": seed + 200,
            }
            for seed in range(8)
        ],
        "kind": "M-noise-floor",
        "metric_ceilings": {"accuracy": 0.01},
        "metric_deltas": {"accuracy": [0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.001]},
        "minimum_signal_threshold": {"accuracy": 0.01},
        "seed_count": 8,
        "seed_ids": list(range(8)),
        "zero_tolerance": 1e-12,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_m_regime_smoke_writes_baseline_artifacts_and_comparison(tmp_path: Path) -> None:
    # Given: a valid noise-floor artifact and tiny smoke config.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        noise_floor_path=noise_floor_path,
        preregistered=True,
    )

    # When: the M-regime smoke gate runs.
    result = run_m_regime_smoke(config)

    # Then: all declared baseline artifacts exist but remain unciteable smoke.
    assert result.comparison.interpretable is False
    assert result.comparison.reason == "below_minimum_validity"
    assert result.status == "blocked"
    assert result.authorize_avsb is False
    assert {run.name for run in result.run_dirs} == set(BASELINE_NAMES)
    comparison = json.loads((config.out_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["interpretable"] is False
    assert comparison["reason"] == "below_minimum_validity"
    for run_dir in result.run_dirs:
        resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        assert resolved["tier"] == "smoke"
        assert resolved["unciteable"] is True
        assert (run_dir / "eval_tensors.json").exists()
        assert (run_dir / "metrics.json").exists()


def test_m_regime_smoke_blocks_without_noise_floor(tmp_path: Path) -> None:
    # Given: a tiny smoke config with no M-noise-floor evidence.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime")

    # When: the M-regime smoke gate runs.
    result = run_m_regime_smoke(config)

    # Then: comparison exists but cannot authorize A-vs-B work.
    assert result.comparison.interpretable is False
    assert result.comparison.reason == "noise_floor_missing"
    assert result.status == "blocked"
    assert result.authorize_avsb is False
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["authorize_avsb"] is False


def test_m_regime_smoke_blocks_when_noise_floor_omits_loss_metric(tmp_path: Path) -> None:
    # Given: a schema-valid preregistered noise floor that omits loss evidence.
    noise_floor_path = _write_accuracy_only_noise_floor(tmp_path / "noise-floor.json")
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        noise_floor_path=noise_floor_path,
        preregistered=True,
    )

    # When: the M-regime smoke gate runs.
    result = run_m_regime_smoke(config)

    # Then: the gate blocks instead of crashing or authorizing A-vs-B.
    assert result.comparison.interpretable is False
    assert result.comparison.reason == "below_minimum_validity"
    assert result.status == "blocked"
    assert result.authorize_avsb is False


def test_m_regime_smoke_records_matched_budget_metadata(tmp_path: Path) -> None:
    # Given: a tiny smoke config with a matched read budget.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", top_k_blocks=2)

    # When: the M-regime smoke gate writes baseline artifacts.
    result = run_m_regime_smoke(config)

    # Then: sparse and Grip variants use the same read budget while dense records none.
    metadata_by_name = {
        run_dir.name: json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        for run_dir in result.run_dirs
    }
    assert metadata_by_name["dense"]["read_budget"] is None
    assert metadata_by_name["local"]["read_budget"] == 2
    assert metadata_by_name["content-sparse"]["read_budget"] == 2
    assert metadata_by_name["grip-read-A"]["read_budget"] == 2
    assert metadata_by_name["grip-select-B"]["read_budget"] == 2
    assert metadata_by_name["local"]["model"]["attention_mode"] == "local"
    assert metadata_by_name["content-sparse"]["model"]["attention_mode"] == "content_sparse"
    assert metadata_by_name["grip-read-A"]["model"]["attention_mode"] == "grip_read"
    assert metadata_by_name["grip-select-B"]["model"]["attention_mode"] == "grip_select"


def test_m_regime_smoke_writes_report_only_selection_diagnostics(tmp_path: Path) -> None:
    # Given: a tiny smoke config with sparse baselines.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", block_size=2, top_k_blocks=2)

    # When: the M-regime gate writes baseline artifacts.
    result = run_m_regime_smoke(config)

    # Then: sparse baselines get report-only selection diagnostics from eval latents.
    diagnostics_by_name = {}
    for run_dir in result.run_dirs:
        diagnostics_path = run_dir / "selection_diagnostics.json"
        if run_dir.name == "dense":
            assert not diagnostics_path.exists()
        else:
            diagnostics_by_name[run_dir.name] = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
    assert diagnostics_by_name["local"]["attention_mode"] == "local"
    assert diagnostics_by_name["local"]["selection_consumed"] is False
    assert diagnostics_by_name["local"]["block_size"] == 2
    assert diagnostics_by_name["local"]["read_budget"] == 2
    assert diagnostics_by_name["content-sparse"]["attention_mode"] == "content_sparse"
    assert diagnostics_by_name["content-sparse"]["selection_consumed"] is True
    assert diagnostics_by_name["content-sparse"]["block_size"] == 2
    assert diagnostics_by_name["content-sparse"]["read_budget"] == 2
    assert isinstance(diagnostics_by_name["content-sparse"]["decisive_token_count"], int)
    assert 0.0 <= diagnostics_by_name["content-sparse"]["decisive_token_recall"] <= 1.0
    assert diagnostics_by_name["grip-read-A"]["attention_mode"] == "grip_read"
    assert diagnostics_by_name["grip-read-A"]["selection_consumed"] is True
    assert diagnostics_by_name["grip-select-B"]["attention_mode"] == "grip_select"
    assert diagnostics_by_name["grip-select-B"]["selection_consumed"] is True
    assert result.authorize_avsb is False


def test_m_regime_smoke_uses_matched_initialization_in_run_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: same-shape sparse baselines observed through the actual run path.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime")
    original_build_model = headroom_runs._build_model
    captured: dict[str, dict[str, torch.Tensor]] = {}

    def capture_build_model(
        build_config: MRegimeConfig,
        spec: BaselineSpec,
    ) -> BaselineModel:
        model = original_build_model(build_config, spec)
        if spec.name in {"local", "content-sparse", "grip-read-A", "grip-select-B"}:
            captured[spec.name] = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        return model

    monkeypatch.setattr(headroom_runs, "_build_model", capture_build_model)

    # When: the full headroom gate constructs all baselines.
    run_m_regime_smoke(config)

    # Then: common parameters start from identical seeded values in that path.
    assert set(captured) == {"local", "content-sparse", "grip-read-A", "grip-select-B"}
    local_state = captured["local"]
    content_sparse_state = captured["content-sparse"]
    shared_keys = local_state.keys() & content_sparse_state.keys()
    assert shared_keys
    for key in shared_keys:
        assert torch.equal(local_state[key], content_sparse_state[key])


def test_m_regime_trained_run_records_training_budget(tmp_path: Path) -> None:
    # Given: a valid noise floor and a tiny trained M-regime config.
    noise_floor_path = _write_noise_floor(
        tmp_path / "noise-floor.json",
        train_steps=1,
        train_batch_size=2,
    )
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        noise_floor_path=noise_floor_path,
        preregistered=True,
        train_steps=1,
        train_batch_size=2,
        lr=1e-3,
    )

    # When: the M-regime gate runs with one training step.
    result = run_m_regime_smoke(config)

    # Then: every baseline records matched training budget and remains scorer-owned but smoke-tier.
    assert result.comparison.interpretable is False
    assert result.comparison.reason == "below_minimum_validity"
    for run_dir in result.run_dirs:
        resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        train_log = (run_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert resolved["train"]["steps"] == 1
        assert resolved["train"]["batch_size"] == 2
        assert resolved["train"]["lr"] == 1e-3
        assert resolved["data"]["task"] == "bayesian"
        assert resolved["run"]["device"] == "cpu"
        assert resolved["tier"] == "smoke"
        assert resolved["unciteable"] is True
        record = json.loads(train_log[-1])
        assert record["step"] == 1
        assert record["seed"] == 0


def test_m_regime_trained_run_uses_heldout_eval_batch(tmp_path: Path) -> None:
    # Given: a trained M-regime config with explicit heldout eval provenance.
    noise_floor_path = _write_noise_floor(
        tmp_path / "noise-floor.json",
        train_steps=1,
        train_batch_size=2,
        eval_batch_size=3,
    )
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        noise_floor_path=noise_floor_path,
        preregistered=True,
        seed=5,
        train_steps=1,
        train_batch_size=2,
        eval_batch_size=3,
        eval_seed_offset=10_000,
    )

    # When: the M-regime gate writes baseline artifacts.
    result = run_m_regime_smoke(config)

    # Then: evaluation is held out from the training batch and recorded.
    dense_dir = next(run_dir for run_dir in result.run_dirs if run_dir.name == "dense")
    resolved = json.loads((dense_dir / "config.resolved.json").read_text(encoding="utf-8"))
    eval_tensors = json.loads((dense_dir / "eval_tensors.json").read_text(encoding="utf-8"))
    assert resolved["eval"] == {
        "batch_size": 3,
        "seed": 10_005,
        "seed_offset": 10_000,
    }
    assert eval_tensors["tokens"] == 24.0
    assert eval_tensors["seed"] == 10_005
    assert eval_tensors["batch_size"] == 3


def test_m_regime_trained_run_requests_disjoint_eval_seed(
    tmp_path: Path,
) -> None:
    # Given: a headroom run with multiple deterministic training batches.
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        seed=3,
        train_steps=2,
        train_batch_size=2,
        eval_batch_size=1,
        eval_seed_offset=10_000,
    )

    # When: the M-regime gate runs.
    result = run_m_regime_smoke(config)

    # Then: train and eval artifacts record disjoint deterministic seeds.
    dense_dir = next(run_dir for run_dir in result.run_dirs if run_dir.name == "dense")
    train_records = [
        json.loads(line)
        for line in (dense_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_tensors = json.loads((dense_dir / "eval_tensors.json").read_text(encoding="utf-8"))
    assert [record["seed"] for record in train_records] == [3_000_000, 3_000_001]
    assert eval_tensors["seed"] == 10_003


def test_m_regime_trained_run_rejects_invalid_training_budget(tmp_path: Path) -> None:
    # Given: a training config with impossible batch size.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", train_steps=1, train_batch_size=0)

    # When / Then: the headroom boundary rejects it before writing run claims.
    with pytest.raises(ValueError, match="train_batch_size"):
        run_m_regime_smoke(config)


def test_m_regime_trained_run_rejects_invalid_eval_budget(tmp_path: Path) -> None:
    # Given: a training config with impossible heldout eval batch size.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", eval_batch_size=0)

    # When / Then: the headroom boundary rejects it before writing run claims.
    with pytest.raises(ValueError, match="eval_batch_size"):
        run_m_regime_smoke(config)


def test_m_regime_trained_run_rejects_eval_seed_overlap(tmp_path: Path) -> None:
    # Given: an eval seed offset that overlaps a deterministic training step seed.
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        seed=0,
        train_steps=3,
        eval_seed_offset=2,
    )

    # When / Then: the headroom boundary rejects the overlapping split.
    with pytest.raises(ValueError, match="eval_seed_offset"):
        run_m_regime_smoke(config)


def test_m_regime_trained_run_rejects_invalid_task(tmp_path: Path) -> None:
    # Given: a training config with an unsupported task.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", task="synthetic")

    # When / Then: the headroom boundary rejects it before writing run claims.
    with pytest.raises(ValueError, match="task"):
        run_m_regime_smoke(config)
