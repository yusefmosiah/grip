"""Smoke tests — must pass before any real run.

Each test corresponds to one milestone's acceptance gate. Run: pytest -q.
"""
import pytest
import torch

from grip.analysis.probe import run_probe_experiment
from grip.data import BayesianEvidenceStream
from grip.models import DenseTransformer


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_available():
    """M-regime prerequisite: MPS is live on this box."""
    assert torch.backends.mps.is_available(), "MPS backend not available"


def test_fixed_shape_contract():
    """The data generator must never emit variable lengths (pytorch #181213)."""
    stream = BayesianEvidenceStream(seq_len=32, vocab_size=16, seed=0)
    first = stream.generate(seed=1)
    second = stream.generate(seed=2)

    assert first.tokens.shape == (32,)
    assert second.tokens.shape == (32,)


def test_probe_runs():
    """Gating experiment #0: the probe executes end-to-end on a toy."""
    stream = BayesianEvidenceStream(num_hypotheses=3, seq_len=16, vocab_size=16, seed=0)
    model = DenseTransformer(
        vocab_size=stream.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=stream.T,
        n_hypotheses=stream.K,
    )

    result = run_probe_experiment(
        model,
        stream,
        n_train_streams=2,
        n_test_streams=1,
        device="cpu",
    )

    assert result.n_train_streams == 2
    assert result.n_test_streams == 1
    assert set(result.level) == {"topmass", "entropy"}
    assert set(result.derivative) == {"d_conf", "dd_conf"}
