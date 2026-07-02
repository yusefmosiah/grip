from __future__ import annotations

import json
from pathlib import Path

import pytest

from grip.eval.headroom import MRegimeConfig, run_m_regime_smoke
from grip.eval.noise_floor import load_noise_floor
from grip.eval.noise_floor_calibration import (
    RIGHT_CALIBRATION_SEED_OFFSET,
    NoiseFloorCalibrationConfig,
    NoiseFloorCalibrationError,
    calibrate_noise_floor,
    main,
)


def test_calibrate_noise_floor_writes_scorer_loadable_loss_artifact(tmp_path: Path) -> None:
    # Given: a minimal Bayesian calibration request with the required seed floor.
    result = calibrate_noise_floor(
        NoiseFloorCalibrationConfig(out_dir=tmp_path / "noise-floor")
    )

    # Then: the artifact is loadable and covers the scorer-owned loss metric.
    artifact = load_noise_floor(result.path)
    raw_artifact = json.loads(result.path.read_text(encoding="utf-8"))
    assert artifact.seed_count == 8
    assert artifact.seed_ids == tuple(range(10_000_000_000, 10_000_000_008))
    assert len(artifact.calibration_pairs) == 8
    assert all(
        pair["left_seed"] != pair["right_seed"]
        for pair in artifact.calibration_pairs
    )
    assert set(artifact.metric_deltas) == {"loss"}
    assert artifact.metric_ceilings["loss"] == max(
        abs(delta) for delta in artifact.metric_deltas["loss"]
    )
    assert artifact.metric_ceilings["loss"] > artifact.zero_tolerance
    assert artifact.minimum_signal_threshold["loss"] == artifact.metric_ceilings["loss"]
    assert raw_artifact["calibration"]["data"] == {
        "seq_len": 8,
        "task": "bayesian",
        "vocab_size": 17,
    }
    assert raw_artifact["calibration"]["eval"] == {
        "batch_size": 1,
        "seed_offset": 10_000,
    }
    assert raw_artifact["calibration"]["model"] == {
        "d_model": 16,
        "n_heads": 4,
        "n_hypotheses": 3,
        "n_layers": 1,
    }
    assert raw_artifact["calibration"]["sparse"] == {
        "block_size": 2,
        "top_k_blocks": 3,
        "window": 2,
    }
    assert raw_artifact["calibration"]["train"] == {
        "batch_size": 1,
        "lr": 1e-3,
        "steps": 0,
    }
    assert raw_artifact["zero_tolerance"] == 1e-6

    # And: the generated artifact can authorize interpretation through compare().
    gate = run_m_regime_smoke(
        MRegimeConfig(
            out_dir=tmp_path / "gate",
            noise_floor_path=result.path,
            preregistered=True,
        )
    )
    assert gate.comparison.interpretable is True
    assert gate.comparison.reason == "ok"


def test_calibrate_noise_floor_rejects_too_few_seeds(tmp_path: Path) -> None:
    # Given: a calibration config below the scorer seed floor.
    config = NoiseFloorCalibrationConfig(out_dir=tmp_path / "noise-floor", seed_ids=tuple(range(7)))

    # When / Then: calibration rejects it before writing an artifact.
    with pytest.raises(NoiseFloorCalibrationError, match="seed_ids"):
        calibrate_noise_floor(config)


def test_calibrate_noise_floor_rejects_duplicate_metric_names(tmp_path: Path) -> None:
    # Given: a calibration config with ambiguous metric ownership.
    config = NoiseFloorCalibrationConfig(
        out_dir=tmp_path / "noise-floor",
        metric_names=("loss", "loss"),
    )

    # When / Then: calibration rejects it before scorer loading.
    with pytest.raises(NoiseFloorCalibrationError, match="metric_names"):
        calibrate_noise_floor(config)


def test_calibrate_noise_floor_rejects_decision_seed_overlap(tmp_path: Path) -> None:
    # Given: a calibration config whose left calibration seed is also a decision seed.
    config = NoiseFloorCalibrationConfig(
        out_dir=tmp_path / "noise-floor",
        seed_ids=tuple(range(1_000_000, 1_000_008)),
        decision_seed_ids=(1_000_000,),
    )

    # When / Then: calibration rejects the overlap before writing artifacts.
    with pytest.raises(NoiseFloorCalibrationError, match="disjoint"):
        calibrate_noise_floor(config)


def test_calibrate_noise_floor_rejects_right_seed_left_seed_collision(tmp_path: Path) -> None:
    # Given: one right-side calibration seed collides with another pair's left seed.
    base = 10_000_000_000
    config = NoiseFloorCalibrationConfig(
        out_dir=tmp_path / "noise-floor",
        seed_ids=(
            base,
            base + RIGHT_CALIBRATION_SEED_OFFSET,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
        ),
    )

    # When / Then: calibration rejects it before pair generation.
    with pytest.raises(NoiseFloorCalibrationError, match="unique"):
        calibrate_noise_floor(config)


def test_calibrate_noise_floor_rejects_right_seed_decision_overlap(tmp_path: Path) -> None:
    # Given: a right-side calibration seed collides with a decision seed.
    base = 10_000_000_000
    config = NoiseFloorCalibrationConfig(
        out_dir=tmp_path / "noise-floor",
        seed_ids=tuple(base + seed for seed in range(8)),
        decision_seed_ids=(base + RIGHT_CALIBRATION_SEED_OFFSET,),
    )

    # When / Then: calibration rejects it before pair generation.
    with pytest.raises(NoiseFloorCalibrationError, match="disjoint"):
        calibrate_noise_floor(config)


def test_calibrate_noise_floor_rejects_degenerate_metric_spread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a calibration run whose paired metrics are exactly identical.
    def zero_pair_delta(left_dir: Path, right_dir: Path, metric: str) -> float:
        return 0.0

    monkeypatch.setattr(
        "grip.eval.noise_floor_calibration._pair_delta",
        zero_pair_delta,
    )

    # When / Then: calibration fails loudly instead of writing an epsilon floor.
    with pytest.raises(NoiseFloorCalibrationError, match="no measurable calibration spread"):
        calibrate_noise_floor(NoiseFloorCalibrationConfig(out_dir=tmp_path / "noise-floor"))


def test_calibrate_noise_floor_rejects_invalid_task_before_writing(tmp_path: Path) -> None:
    # Given: a calibration config outside the supported task set.
    out_dir = tmp_path / "noise-floor"
    config = NoiseFloorCalibrationConfig(out_dir=out_dir, task="synthetic")

    # When / Then: calibration rejects it before creating artifacts.
    with pytest.raises(NoiseFloorCalibrationError, match="task"):
        calibrate_noise_floor(config)
    assert not out_dir.exists()


def test_noise_floor_calibration_cli_writes_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Given: CLI arguments for a minimal calibration run.
    out_dir = tmp_path / "cli-noise-floor"

    # When: the calibration CLI runs.
    exit_code = main([str(out_dir), "--seed-count", "8"])

    # Then: it prints a loadable artifact path.
    assert exit_code == 0
    printed_path = Path(capsys.readouterr().out.strip())
    assert printed_path == out_dir / "noise-floor.json"
    assert load_noise_floor(printed_path).seed_count == 8


def test_noise_floor_calibration_cli_records_heldout_eval_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: CLI arguments with explicit heldout eval options.
    out_dir = tmp_path / "cli-noise-floor"

    # When: the calibration CLI runs.
    exit_code = main(
        [
            str(out_dir),
            "--seed-count",
            "8",
            "--eval-batch-size",
            "2",
            "--eval-seed-offset",
            "20000",
        ]
    )

    # Then: the calibration artifact records the heldout eval policy.
    assert exit_code == 0
    printed_path = Path(capsys.readouterr().out.strip())
    payload = json.loads(printed_path.read_text(encoding="utf-8"))
    assert payload["calibration"]["eval"] == {
        "batch_size": 2,
        "seed_offset": 20_000,
    }
