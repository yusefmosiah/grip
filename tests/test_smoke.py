"""Smoke tests — must pass before any real run.

Each test corresponds to one milestone's acceptance gate. Run: pytest -q.
"""
import pytest
import torch


def test_mps_available():
    """M-regime prerequisite: MPS is live on this box."""
    assert torch.backends.mps.is_available(), "MPS backend not available"


def test_fixed_shape_contract():
    """The data generator must never emit variable lengths (pytorch #181213)."""
    # CODEX: instantiate BayesianEvidenceStream(seq_len=T), generate() twice,
    # assert both .tokens.shape == (T,). Non-negotiable.
    pytest.skip("CODEX: fill once streams.py is implemented")


def test_probe_runs():
    """Gating experiment #0: the probe executes end-to-end on a toy."""
    pytest.skip("CODEX: fill once probe.py + a tiny dense model exist")
