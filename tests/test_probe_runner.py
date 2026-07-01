from grip.analysis.probe import ProbeExperimentResult, ProbeResult, run_probe_experiment
from grip.analysis.run_probe_000 import (
    PROBE_TEST_SEED_BASE,
    PROBE_TRAIN_SEED_BASE,
    SEED_STRIDE,
    interpret_probe_result,
    probe_seed_bases,
    train_backbone,
)
from grip.data import BayesianEvidenceStream
from grip.models import DenseTransformer


def _probe_result(name: str, r2: float) -> ProbeResult:
    return ProbeResult(target_name=name, mse=0.0, r2=r2, n_train=10, n_test=5)


def _experiment(topmass_r2: float, entropy_r2: float, d_conf_r2: float) -> ProbeExperimentResult:
    return ProbeExperimentResult(
        level={
            "topmass": _probe_result("topmass", topmass_r2),
            "entropy": _probe_result("entropy", entropy_r2),
        },
        derivative={
            "d_conf": _probe_result("d_conf", d_conf_r2),
            "dd_conf": _probe_result("dd_conf", 0.01),
        },
        routing={},
        n_train_streams=10,
        n_test_streams=5,
        backbone_params=100,
        backbone_seed=0,
        probe_train_seed_base=10_000_000,
        probe_test_seed_base=20_000_000,
    )


def test_interpretation_is_invalid_when_level_control_fails():
    result = interpret_probe_result(_experiment(topmass_r2=0.1, entropy_r2=0.9, d_conf_r2=0.0))

    assert result.status == "invalid_level_control"
    assert not result.level_control_passed
    assert "topmass" in result.message


def test_interpretation_supports_amnesia_only_after_controls_pass():
    result = interpret_probe_result(_experiment(topmass_r2=0.9, entropy_r2=0.8, d_conf_r2=0.0))

    assert result.status == "amnesia_supported"
    assert result.level_control_passed


def test_interpretation_reports_mixed_derivative_readability():
    result = interpret_probe_result(_experiment(topmass_r2=0.9, entropy_r2=0.8, d_conf_r2=0.4))

    assert result.status == "mixed_derivative_result"
    assert result.level_control_passed
    assert "d_conf" in result.message


def test_train_backbone_accepts_level_auxiliary_weights():
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=8, vocab_size=8, seed=0)

    model = train_backbone(
        stream,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_steps=1,
        batch=1,
        device="cpu",
        lm_weight=0.1,
        aux_weight=1.0,
        topmass_weight=1.0,
        entropy_weight=1.0,
        log_every=99,
    )

    assert isinstance(model, DenseTransformer)


def test_probe_experiment_accepts_disjoint_probe_seed_ranges():
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=8, vocab_size=8, seed=0)
    model = DenseTransformer(
        vocab_size=stream.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=stream.T,
        n_hypotheses=stream.K,
    )

    result = run_probe_experiment(
        model,
        stream,
        n_train_streams=2,
        n_test_streams=2,
        device="cpu",
        probe_epochs=1,
        probe_train_seed_base=10_000_000,
        probe_test_seed_base=20_000_000,
    )

    assert result.probe_train_seed_base == 10_000_000
    assert result.probe_test_seed_base == 20_000_000


def test_default_runner_probe_seed_ranges_do_not_overlap_training_ranges():
    for seed in range(3):
        train_base, test_base = probe_seed_bases(seed)
        seed_seq = seed * 13 + 1
        online_training_first = seed_seq * 1000
        online_training_last = (seed_seq + 1500 - 1) * 1000 + 8 - 1

        assert train_base == PROBE_TRAIN_SEED_BASE + seed * SEED_STRIDE
        assert test_base == PROBE_TEST_SEED_BASE + seed * SEED_STRIDE
        assert train_base > online_training_last
        assert test_base > train_base + 200
