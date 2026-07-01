from __future__ import annotations

import numpy as np
import pytest

from grip.analysis.bypass import (
    BypassProbeConfig,
    BypassProbeConfigError,
    collect_bypass_dataset,
    run_bypass_probe,
)
from grip.analysis.run_bypass_002 import BypassRunConfig, run_gate
from grip.data import BayesianEvidenceStream, StreamSample


class TokenLeakStream:
    T = 12
    K = 2
    S = 1
    vocab_size = 4

    def generate(self, seed: int | None = None) -> StreamSample:
        sample_seed = 0 if seed is None else seed
        answer = sample_seed % self.K
        tokens = np.full(self.T, answer + 1, dtype=np.int64)
        posterior = np.full((self.T, self.K), 0.5, dtype=np.float64)
        posterior[:, answer] = 0.95
        posterior[:, 1 - answer] = 0.05
        d_conf = np.zeros(self.T, dtype=np.float64)
        d_conf[1:] = 1.0 if answer == 1 else -1.0
        return StreamSample(
            tokens=tokens,
            answer=answer,
            posterior=posterior,
            entropy=np.zeros(self.T, dtype=np.float64),
            belief_move=d_conf.copy(),
            d_conf=d_conf,
            dd_conf=np.zeros(self.T, dtype=np.float64),
            source_idx=np.zeros(self.T, dtype=np.int64),
            source_trust=np.ones((self.T, self.S), dtype=np.float64),
            decisive_idx=(np.abs(d_conf) > 0).astype(np.int64),
            block_boundaries=np.asarray([0, self.T], dtype=np.int64),
            metadata={"natural_len": self.T},
        )


class OrderLeakStream:
    T = 12
    K = 2
    S = 1
    vocab_size = 4

    def generate(self, seed: int | None = None) -> StreamSample:
        sample_seed = 0 if seed is None else seed
        answer = sample_seed % self.K
        if answer == 0:
            pattern = [1, 2] * (self.T // 2)
        else:
            pattern = [2, 1] * (self.T // 2)
        tokens = np.asarray(pattern, dtype=np.int64)
        posterior = np.full((self.T, self.K), 0.5, dtype=np.float64)
        posterior[:, answer] = 0.95
        posterior[:, 1 - answer] = 0.05
        d_conf = np.zeros(self.T, dtype=np.float64)
        return StreamSample(
            tokens=tokens,
            answer=answer,
            posterior=posterior,
            entropy=np.zeros(self.T, dtype=np.float64),
            belief_move=d_conf.copy(),
            d_conf=d_conf,
            dd_conf=np.zeros(self.T, dtype=np.float64),
            source_idx=np.zeros(self.T, dtype=np.int64),
            source_trust=np.ones((self.T, self.S), dtype=np.float64),
            decisive_idx=np.zeros(self.T, dtype=np.int64),
            block_boundaries=np.asarray([0, self.T], dtype=np.int64),
            metadata={"natural_len": self.T},
        )


def test_collect_bypass_dataset_uses_disjoint_seed_ranges():
    # Given: a real stream and explicit train/test seed bases.
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=16, vocab_size=8, seed=0)
    config = BypassProbeConfig(
        n_train_streams=3,
        n_test_streams=2,
        train_seed_base=100,
        test_seed_base=200,
        window=3,
        probe_epochs=1,
    )

    # When: the raw-token bypass dataset is collected.
    dataset = collect_bypass_dataset(stream, config)

    # Then: split counts and seed ranges remain explicit and disjoint.
    assert dataset.n_train_streams == 3
    assert dataset.n_test_streams == 2
    assert dataset.train_seed_base == 100
    assert dataset.test_seed_base == 200
    assert dataset.train_position_mask.any()
    assert (~dataset.train_position_mask).any()


def test_collect_bypass_dataset_rejects_overlapping_seed_ranges():
    # Given: train and test generated-seed ranges that overlap.
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=16, vocab_size=8, seed=0)
    config = BypassProbeConfig(
        n_train_streams=10,
        n_test_streams=10,
        train_seed_base=100,
        test_seed_base=105,
    )

    # When / Then: collection rejects the leaky split.
    with pytest.raises(BypassProbeConfigError, match="seed ranges"):
        collect_bypass_dataset(stream, config)


def test_bypass_probe_flags_legible_token_leak():
    # Given: a stream whose raw tokens directly encode both answer and d_conf.
    stream = TokenLeakStream()
    config = BypassProbeConfig(
        n_train_streams=16,
        n_test_streams=8,
        train_seed_base=0,
        test_seed_base=100,
        window=2,
        probe_epochs=200,
        d_conf_r2_threshold=0.2,
        answer_acc_threshold=0.9,
    )

    # When: the bypass probe runs.
    result = run_bypass_probe(stream, config)

    # Then: the gate rejects the task as raw-token legible.
    assert result.d_conf_r2 > 0.9
    assert result.answer_accuracy > 0.9
    assert not result.d_conf_passed
    assert not result.answer_passed
    assert not result.passed


def test_bypass_probe_flags_order_only_answer_leak():
    # Given: a stream where token counts are identical and only order leaks answer.
    stream = OrderLeakStream()
    config = BypassProbeConfig(
        n_train_streams=20,
        n_test_streams=8,
        train_seed_base=0,
        test_seed_base=100,
        window=2,
        probe_epochs=200,
        answer_acc_threshold=0.9,
    )

    # When: the bypass probe runs on raw positional token features.
    result = run_bypass_probe(stream, config)

    # Then: order-only raw-token answer leakage is rejected.
    assert result.answer_accuracy > 0.9
    assert not result.answer_passed


def test_bypass_probe_reports_gate_decision_for_bayesian_stream():
    # Given: the real Bayesian evidence stream.
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=32, vocab_size=16, seed=7)
    config = BypassProbeConfig(
        n_train_streams=12,
        n_test_streams=6,
        train_seed_base=1_000,
        test_seed_base=2_000,
        window=4,
        probe_epochs=20,
    )

    # When: the M-legibility gate runs.
    result = run_bypass_probe(stream, config)

    # Then: it reports measured metrics and thresholded pass/fail decisions.
    assert result.n_train_positions > 0
    assert result.n_test_positions > 0
    assert isinstance(result.d_conf_passed, bool)
    assert isinstance(result.answer_passed, bool)
    assert result.passed == (result.d_conf_passed and result.answer_passed)


def test_bypass_runner_writes_report(tmp_path):
    # Given: a tiny M-legibility run configuration.
    config = BypassRunConfig(
        out_dir=str(tmp_path),
        seq_len=24,
        num_hypotheses=2,
        num_sources=1,
        vocab_size=12,
        probe=BypassProbeConfig(
            n_train_streams=4,
            n_test_streams=2,
            train_seed_base=10,
            test_seed_base=20,
            window=3,
            probe_epochs=2,
        ),
    )

    # When: the runner executes the gate.
    report = run_gate(config)

    # Then: the report is both returned and written as the durable artifact.
    assert report["gate"] == "M-legibility"
    assert (tmp_path / "report.json").exists()
    assert report["probe_config"]["n_train_streams"] == 4
    assert "d_conf_r2" in report["metrics"]
    assert set(report["thresholds"]) == {"d_conf_r2_threshold", "answer_acc_threshold"}
    assert set(report["decision"]) == {"d_conf_passed", "answer_passed", "passed"}
    assert report["decision"]["passed"] == (
        report["decision"]["d_conf_passed"] and report["decision"]["answer_passed"]
    )
