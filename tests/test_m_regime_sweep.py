from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.m_regime_sweep import (
    MRegimeSweepConfig,
    _calibration_config,
    _losses,
    main,
    run_m_regime_sweep,
)
from grip.eval.headroom_types import MRegimeResult
from grip.eval.score import score_run
from grip.eval.score_types import ComparisonReport

from score_fixtures import write_run


def _write_named_run(run_dir: Path, model_name: str, *, loss: float) -> Path:
    write_run(run_dir, {"loss": loss})
    resolved_path = run_dir / "config.resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["model"]["name"] = model_name
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def test_m_regime_sweep_writes_summary_and_aggregate_reports(tmp_path: Path) -> None:
    # Given: a tiny calibrated Bayesian M-regime sweep config.
    config = MRegimeSweepConfig(out_dir=tmp_path / "sweep")

    # When: the sweep runner executes the calibrated per-seed path.
    result = run_m_regime_sweep(config)

    # Then: it writes the calibrated floor, per-seed summary, and aggregate reports.
    assert result.noise_floor_path.exists()
    assert result.summary_path == config.out_dir / "summary.json"
    assert result.summary_path.exists()
    assert result.aggregate.report_path == config.out_dir / "aggregate" / "aggregate-summary.json"
    assert result.aggregate.report_path.exists()
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["bayesian"]["config"] == {
            "data": {
                "seq_len": 8,
                "task": "bayesian",
                "vocab_size": 17,
            },
            "decision": {
                "seed_count": 8,
            },
            "eval": {
            "batch_size": 1,
            "seed_offset": 10_000,
        },
        "model": {
            "d_model": 16,
            "n_heads": 4,
            "n_hypotheses": 3,
            "n_layers": 1,
        },
        "sparse": {
            "block_size": 2,
            "top_k_blocks": 3,
            "window": 2,
        },
        "train": {
            "batch_size": 1,
            "lr": 1e-3,
            "steps": 0,
        },
    }
    rows = payload["bayesian"]["rows"]
    assert len(rows) == 8
    assert all(row["interpretable"] is False for row in rows)
    assert all(row["reason"] == "below_minimum_validity" for row in rows)
    assert all(row["tier"] == "smoke" for row in rows)
    assert all(row["unciteable"] is True for row in rows)
    diagnostics = rows[0]["selection_diagnostics"]
    assert diagnostics["local"]["attention_mode"] == "local"
    assert diagnostics["local"]["selection_consumed"] is False
    assert diagnostics["content-sparse"]["attention_mode"] == "content_sparse"
    assert diagnostics["content-sparse"]["selection_consumed"] is True
    assert 0.0 <= diagnostics["content-sparse"]["decisive_token_recall"] <= 1.0
    assert result.aggregate.tasks[0].task == "bayesian"
    assert result.aggregate.tasks[0].decision.authorize_avsb is False


def test_sweep_losses_use_resolved_model_names_not_run_dir_names(tmp_path: Path) -> None:
    run_dirs = (
        _write_named_run(tmp_path / "path-a", "content-sparse", loss=0.30),
        _write_named_run(tmp_path / "path-b", "dense", loss=0.20),
        _write_named_run(tmp_path / "path-c", "local", loss=0.25),
    )
    result = MRegimeResult(
        run_dirs=run_dirs,
        comparison=ComparisonReport(
            runs=tuple(score_run(run_dir) for run_dir in run_dirs),
            interpretable=False,
            reason="test",
            noise_floor=None,
        ),
        report_path=tmp_path / "report.json",
        status="blocked",
        authorize_avsb=False,
    )

    losses = _losses(result)

    assert losses.dense == 0.20
    assert losses.local == 0.25
    assert losses.content_sparse == 0.30


def test_m_regime_sweep_cli_prints_aggregate_summary_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a tiny output directory for the reusable sweep CLI.
    out_dir = tmp_path / "cli-sweep"

    # When: the CLI executes the default calibrated Bayesian sweep.
    exit_code = main([str(out_dir)])

    # Then: it prints the aggregate-summary path and writes the expected artifacts.
    assert exit_code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(out_dir / "aggregate" / "aggregate-summary.json")
    assert (out_dir / "noise-floor" / "noise-floor.json").exists()
    assert (out_dir / "summary.json").exists()
    assert Path(printed).exists()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(payload["bayesian"]["rows"]) == 8
    resolved = json.loads(
        (out_dir / "decisions" / "seed-0" / "dense" / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    assert resolved["eval"]["seed"] == 10_000
    assert resolved["eval"]["batch_size"] == 1


def test_m_regime_sweep_cli_propagates_heldout_eval_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a tiny output directory and explicit heldout eval options.
    out_dir = tmp_path / "cli-sweep"

    # When: the CLI executes the default calibrated Bayesian sweep.
    exit_code = main(
        [
            str(out_dir),
            "--eval-batch-size",
            "2",
            "--eval-seed-offset",
            "20000",
        ]
    )

    # Then: downstream decision artifacts record the requested eval policy.
    assert exit_code == 0
    capsys.readouterr()
    resolved = json.loads(
        (out_dir / "decisions" / "seed-0" / "dense" / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    assert resolved["eval"]["seed"] == 20_000
    assert resolved["eval"]["batch_size"] == 2


def test_m_regime_sweep_calibration_seeds_do_not_overlap_decision_training_space(tmp_path: Path) -> None:
    # Given: adjacent decision seeds whose first training batches used to collide with calibration seeds.
    config = MRegimeSweepConfig(
        out_dir=tmp_path / "sweep",
        seed_ids=tuple(range(8)),
        train_steps=1,
    )

    # When: the sweep builds its calibration config.
    calibration = _calibration_config(config)

    # Then: calibration seeds are outside the decision seed/training/eval ranges.
    assert set(calibration.seed_ids).isdisjoint(config.seed_ids)
    assert set(calibration.seed_ids).isdisjoint(seed * 1_000_000 for seed in config.seed_ids)
    assert set(calibration.seed_ids).isdisjoint(seed + config.eval_seed_offset for seed in config.seed_ids)


def test_m_regime_sweep_rejects_reversal_shape_before_artifacts(tmp_path: Path) -> None:
    # Given: an invalid reversal sweep shape.
    config = MRegimeSweepConfig(
        out_dir=tmp_path / "sweep",
        task="reversal",
        seq_len=8,
        vocab_size=64,
        n_hypotheses=4,
    )

    # When / Then: calibration validation rejects it before writing artifacts.
    with pytest.raises(ValueError, match="seq_len"):
        run_m_regime_sweep(config)
    assert not config.out_dir.exists()


def test_m_regime_sweep_cli_rejects_seed_count_before_artifacts(tmp_path: Path) -> None:
    # Given: an output directory and a seed count below the M-noise-floor floor.
    out_dir = tmp_path / "cli-sweep"

    # When / Then: argparse rejects it before domain execution writes artifacts.
    with pytest.raises(SystemExit) as exc_info:
        main([str(out_dir), "--seed-count", "7"])
    assert exc_info.value.code == 2
    assert not out_dir.exists()
