from __future__ import annotations

from grip.eval.m_regime_validity import noise_floor_validity, run_tier, run_validity


def _valid_payload() -> dict:
    return {
        "decision": {"seed_count": 8},
        "data": {"seq_len": 512},
        "eval": {"batch_size": 8},
        "train": {"batch_size": 8, "steps": 1000},
    }


def test_run_validity_accepts_amendment_minimums() -> None:
    # Given: a payload at every SPEC-002-AMENDMENT-001 minimum.
    payload = _valid_payload()

    # Then: it is a valid-tier M-regime cell.
    assert run_validity(payload) == ()
    assert run_tier(payload) == "valid"


def test_run_validity_rejects_each_below_minimum_field() -> None:
    # Given: a payload below every SPEC-002-AMENDMENT-001 minimum.
    payload = {
        "decision": {"seed_count": 7},
        "data": {"seq_len": 511},
        "eval": {"batch_size": 7},
        "train": {"batch_size": 7, "steps": 999},
    }

    # Then: every missing minimum is named.
    assert run_validity(payload) == (
        "decision.seed_count",
        "data.seq_len",
        "eval.batch_size",
        "train.batch_size",
        "train.steps",
    )
    assert run_tier(payload) == "smoke"


def test_noise_floor_validity_requires_eight_calibration_pairs() -> None:
    # Given / When / Then: calibration-pair count is part of the valid-tier gate.
    assert noise_floor_validity(7) == ("noise_floor.calibration_pairs",)
    assert noise_floor_validity(8) == ()
