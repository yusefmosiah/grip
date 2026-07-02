from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.score import score_run
from grip.train.run import ConfigError, DataConfig, ModelConfig, RunConfig, RunSettings, TrainConfig, main, train


def _smoke_config() -> dict:
    return {
        "model": {"name": "dense", "size": "1M", "d_model": 32, "n_layers": 1},
        "data": {"task": "T0-bayesian-evidence-streams", "seq_len": 128, "num_hypotheses": 4},
        "train": {"steps": 0, "microbatch": 1, "grad_accum": 1, "lr": 0.001, "seed": 123},
        "device": "cpu",
        "run": {"mode": "stub-dry-run"},
    }


def test_train_dry_run_writes_deterministic_artifacts(tmp_path: Path) -> None:
    # Given: a minimal smoke config and two empty run directories.
    config = _smoke_config()
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"

    # When: the dry-run artifact path is executed twice.
    assert train(config, run_a) == run_a
    assert train(config, run_b) == run_b

    # Then: reproducibility artifacts are stable across run directories.
    resolved_a = json.loads((run_a / "config.resolved.json").read_text(encoding="utf-8"))
    resolved_b = json.loads((run_b / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved_a == resolved_b
    assert resolved_a["train"]["seed"] == 123
    assert resolved_a["device"] == "cpu"
    assert resolved_a["run"]["mode"] == "stub-dry-run"
    assert resolved_a["model"]["name"] == "dense"
    assert resolved_a["data"]["seq_len"] == 128
    assert resolved_a["read_budget"] is None
    assert (run_a / "train.jsonl").read_text(encoding="utf-8") == (
        run_b / "train.jsonl"
    ).read_text(encoding="utf-8")
    train_record = json.loads((run_a / "train.jsonl").read_text(encoding="utf-8"))
    assert train_record["lr"] == 0.001
    assert train_record["elapsed_seconds"] == 0.0
    assert (run_a / "eval_tensors.json").exists()
    assert not (run_a / "metrics.json").exists()
    assert score_run(run_a).metrics == {"loss": 0.0, "tokens": 0.0}
    assert (run_a / "metrics.json").exists()


def test_train_artifacts_do_not_claim_winners(tmp_path: Path) -> None:
    # Given: a minimal smoke config.
    run_dir = tmp_path / "run"

    # When: the dry-run artifact path is executed.
    train(_smoke_config(), run_dir)

    # Then: trainer-authored artifacts contain no comparison claims.
    joined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [run_dir / "config.resolved.json", run_dir / "train.jsonl", run_dir / "eval_tensors.json"]
    ).lower()
    assert "winner" not in joined
    assert "wins" not in joined
    assert "beats" not in joined


def test_train_cli_reads_json_config_and_writes_run_dir(tmp_path: Path, capsys) -> None:
    # Given: a JSON config file for the fenced stub trainer.
    config_path = tmp_path / "config.json"
    run_dir = tmp_path / "cli-run"
    config_path.write_text(json.dumps(_smoke_config()), encoding="utf-8")

    # When: the train CLI is invoked.
    exit_code = main([str(config_path), str(run_dir)])

    # Then: it writes artifacts and prints the run directory.
    assert exit_code == 0
    assert Path(capsys.readouterr().out.strip()) == run_dir
    assert (run_dir / "config.resolved.json").exists()
    assert score_run(run_dir).metrics == {"loss": 0.0, "tokens": 0.0}


@pytest.mark.parametrize("model_name", ["grip-read-A", "grip-select-B"])
def test_train_stub_accepts_declared_grip_variant_names(tmp_path: Path, model_name: str) -> None:
    # Given: an explicit stub-dry-run config for a declared Grip variant name.
    config = _smoke_config()
    config["model"]["name"] = model_name
    config["model"]["top_k_blocks"] = 2

    # When: the fenced stub path writes artifacts.
    run_dir = train(config, tmp_path / model_name)

    # Then: resolved config preserves the declared variant.
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved["model"]["name"] == model_name


def test_train_rejects_missing_required_config_section(tmp_path: Path) -> None:
    # Given: a config missing the data section.
    config = _smoke_config()
    del config["data"]

    # When / Then: boundary parsing rejects it before writing artifacts.
    with pytest.raises(ConfigError, match="data"):
        train(config, tmp_path / "run")


@pytest.mark.parametrize("mode", ["smoke", "preregistered"])
def test_train_rejects_non_stub_mode_before_writing_artifacts(tmp_path: Path, mode: str) -> None:
    # Given: an official-looking mode pointed at the stub trainer.
    config = _smoke_config()
    config["run"]["mode"] = mode
    run_dir = tmp_path / "run"

    # When / Then: the trainer rejects it before writing stub artifacts.
    with pytest.raises(ConfigError, match="stub-dry-run"):
        train(config, run_dir)
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model", "name", "bogus"),
        ("root", "device", "gpu"),
        ("run", "mode", "final-winner"),
    ],
)
def test_train_rejects_closed_set_config_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
) -> None:
    # Given: a config with a closed-set field set to an unsupported value.
    config = _smoke_config()
    if section == "root":
        config[field] = value
    else:
        config[section][field] = value

    # When / Then: boundary parsing rejects it before writing artifacts.
    with pytest.raises(ConfigError, match=field):
        train(config, tmp_path / "run")


def test_train_revalidates_direct_run_config_input(tmp_path: Path) -> None:
    # Given: a manually constructed RunConfig with invalid closed-set values.
    config = RunConfig(
        model=ModelConfig(name="bogus", size="1M", d_model=32, n_layers=1, top_k_blocks=None),
        data=DataConfig(task="T0-bayesian-evidence-streams", seq_len=128, num_hypotheses=4),
        train=TrainConfig(steps=0, microbatch=1, grad_accum=1, lr=0.001, seed=123),
        device="gpu",
        run=RunSettings(mode="final-winner"),
    )

    # When / Then: train rejects it before writing artifacts.
    with pytest.raises(ConfigError, match="name"):
        train(config, tmp_path / "run")
