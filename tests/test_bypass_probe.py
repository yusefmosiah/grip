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
from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, StreamSample


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
    assert result.answer_acc_threshold == 0.8
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
    assert set(report["metrics"]) == {"T0-bayesian-evidence-streams", "T1-source-reliability-reversal"}
    assert "d_conf_r2" in report["metrics"]["T0-bayesian-evidence-streams"]
    assert "d_conf_r2" in report["metrics"]["T1-source-reliability-reversal"]
    assert set(report["thresholds"]) == {"d_conf_r2_threshold", "answer_acc_threshold"}
    assert set(report["decision"]) == {"d_conf_passed", "answer_passed", "passed"}
    assert report["decision"]["passed"] == (
        report["decision"]["d_conf_passed"] and report["decision"]["answer_passed"]
    )


def test_bypass_runner_writes_t0_and_t1_reports(tmp_path):
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

    # When: the runner executes the bypass gate.
    report = run_gate(config)

    # Then: both T0 and T1 have independent legibility reports.
    assert set(report["tasks"]) == {"T0-bayesian-evidence-streams", "T1-source-reliability-reversal"}
    assert (tmp_path / "T0-bayesian-evidence-streams-report.json").exists()
    assert (tmp_path / "T1-source-reliability-reversal-report.json").exists()


def test_t1_reversal_stream_passes_bypass_gate_after_leakage_hardening():
    # Given: the lead T1 source-reliability reversal task.
    stream = SourceReliabilityReversalStream(seq_len=48, seed=7)
    config = BypassProbeConfig(
        n_train_streams=24,
        n_test_streams=12,
        train_seed_base=1_000,
        test_seed_base=2_000,
        window=4,
        probe_epochs=40,
        answer_acc_threshold=0.45,
    )

    # When: the raw-token bypass gate is run on T1.
    result = run_bypass_probe(stream, config)

    # Then: T1 is held to the same M-legibility standard as T0.
    assert result.passed


def test_t1_reversal_stream_passes_bypass_gate_across_preregistered_seed_count():
    # Given: the lead T1 task under the default M-legibility gate.
    config = BypassProbeConfig()

    # When: the first preregistered eight stream seeds are checked.
    results = [
        run_bypass_probe(SourceReliabilityReversalStream(seq_len=128, seed=seed), config)
        for seed in range(8)
    ]

    # Then: leakage hardening is not dependent on one favorable stream seed.
    assert all(result.passed for result in results)
    assert {result.answer_acc_threshold for result in results} == {0.4}
    assert max(result.answer_accuracy for result in results) <= max(
        result.answer_acc_threshold
        for result in results
    )
    assert max(result.d_conf_r2 for result in results) <= config.d_conf_r2_threshold
