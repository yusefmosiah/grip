from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import grip.eval.headroom as headroom
from grip.eval.headroom import MRegimeConfig, run_m_regime_smoke
from grip.eval.headroom_types import BaselineSpec
from grip.models import ContentSparseTransformer, DenseTransformer


BaselineModel = DenseTransformer | ContentSparseTransformer


def _write_noise_floor(path: Path) -> Path:
    payload = {
        "identical_config_pairs": [
            {"left": f"seed-{seed}-a", "right": f"seed-{seed}-b"}
            for seed in range(8)
        ],
        "kind": "M-noise-floor",
        "metric_ceilings": {"loss": 0.01},
        "metric_deltas": {"loss": [0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.001]},
        "minimum_signal_threshold": {"loss": 0.01},
        "seed_count": 8,
        "seed_ids": list(range(8)),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_accuracy_only_noise_floor(path: Path) -> Path:
    payload = {
        "identical_config_pairs": [
            {"left": f"seed-{seed}-a", "right": f"seed-{seed}-b"}
            for seed in range(8)
        ],
        "kind": "M-noise-floor",
        "metric_ceilings": {"accuracy": 0.01},
        "metric_deltas": {"accuracy": [0.01, -0.01, 0.0, 0.005, -0.005, 0.002, -0.002, 0.001]},
        "minimum_signal_threshold": {"accuracy": 0.01},
        "seed_count": 8,
        "seed_ids": list(range(8)),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_m_regime_smoke_writes_baseline_artifacts_and_comparison(tmp_path: Path) -> None:
    # Given: a valid preregistered noise-floor artifact and tiny smoke config.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
    config = MRegimeConfig(
        out_dir=tmp_path / "m-regime",
        noise_floor_path=noise_floor_path,
        preregistered=True,
    )

    # When: the M-regime smoke gate runs.
    result = run_m_regime_smoke(config)

    # Then: dense/local/content-sparse artifacts and scorer comparison exist.
    assert result.comparison.interpretable is True
    assert result.status in {"keep", "pivot"}
    assert result.authorize_avsb is (result.status == "keep")
    assert {run.name for run in result.run_dirs} == {"dense", "local", "content-sparse"}
    comparison = json.loads((config.out_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["interpretable"] is True
    for run_dir in result.run_dirs:
        assert (run_dir / "config.resolved.json").exists()
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
    assert result.comparison.reason == "noise_floor_missing_metric"
    assert result.status == "blocked"
    assert result.authorize_avsb is False


def test_m_regime_smoke_records_matched_budget_metadata(tmp_path: Path) -> None:
    # Given: a tiny smoke config with a matched read budget.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", top_k_blocks=2)

    # When: the M-regime smoke gate writes baseline artifacts.
    result = run_m_regime_smoke(config)

    # Then: local and content-sparse use the same read budget while dense records none.
    metadata_by_name = {
        run_dir.name: json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        for run_dir in result.run_dirs
    }
    assert metadata_by_name["dense"]["read_budget"] is None
    assert metadata_by_name["local"]["read_budget"] == 2
    assert metadata_by_name["content-sparse"]["read_budget"] == 2
    assert metadata_by_name["local"]["model"]["attention_mode"] == "local"
    assert metadata_by_name["content-sparse"]["model"]["attention_mode"] == "content_sparse"


def test_m_regime_smoke_uses_matched_initialization_in_run_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: same-shape sparse baselines observed through the actual run path.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime")
    original_build_model = headroom._build_model
    captured: dict[str, dict[str, torch.Tensor]] = {}

    def capture_build_model(
        build_config: MRegimeConfig,
        spec: BaselineSpec,
    ) -> BaselineModel:
        model = original_build_model(build_config, spec)
        if spec.name in {"local", "content-sparse"}:
            captured[spec.name] = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
        return model

    monkeypatch.setattr(headroom, "_build_model", capture_build_model)

    # When: the full headroom gate constructs all baselines.
    run_m_regime_smoke(config)

    # Then: common parameters start from identical seeded values in that path.
    assert set(captured) == {"local", "content-sparse"}
    local_state = captured["local"]
    content_sparse_state = captured["content-sparse"]
    shared_keys = local_state.keys() & content_sparse_state.keys()
    assert shared_keys
    for key in shared_keys:
        assert torch.equal(local_state[key], content_sparse_state[key])


def test_m_regime_trained_run_records_training_budget(tmp_path: Path) -> None:
    # Given: a valid noise floor and a tiny trained M-regime config.
    noise_floor_path = _write_noise_floor(tmp_path / "noise-floor.json")
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

    # Then: every baseline records matched training budget and remains scorer-owned.
    assert result.comparison.interpretable is True
    for run_dir in result.run_dirs:
        resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        train_log = (run_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert resolved["train"]["steps"] == 1
        assert resolved["train"]["batch_size"] == 2
        assert resolved["train"]["lr"] == 1e-3
        assert resolved["data"]["task"] == "bayesian"
        assert resolved["run"]["device"] == "cpu"
        assert json.loads(train_log[-1])["step"] == 1


def test_m_regime_trained_run_rejects_invalid_training_budget(tmp_path: Path) -> None:
    # Given: a training config with impossible batch size.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", train_steps=1, train_batch_size=0)

    # When / Then: the headroom boundary rejects it before writing run claims.
    with pytest.raises(ValueError, match="train_batch_size"):
        run_m_regime_smoke(config)


def test_m_regime_trained_run_rejects_invalid_task(tmp_path: Path) -> None:
    # Given: a training config with an unsupported task.
    config = MRegimeConfig(out_dir=tmp_path / "m-regime", task="synthetic")

    # When / Then: the headroom boundary rejects it before writing run claims.
    with pytest.raises(ValueError, match="task"):
        run_m_regime_smoke(config)
