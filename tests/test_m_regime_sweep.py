from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.m_regime_sweep import MRegimeSweepConfig, run_m_regime_sweep


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
