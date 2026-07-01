from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.m_regime_sweep import MRegimeSweepConfig, main, run_m_regime_sweep


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
    rows = payload["bayesian"]["rows"]
    assert len(rows) == 8
    assert all(row["interpretable"] is True for row in rows)
    assert result.aggregate.tasks[0].task == "bayesian"
    assert result.aggregate.tasks[0].decision.authorize_avsb is False


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
