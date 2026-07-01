import json

import torch
import pytest

from grip.analysis.probe import ProbeExperimentResult, ProbeResult, linear_probe, run_probe_experiment
from grip.analysis.run_probe_000 import (
    PROBE_TEST_SEED_BASE,
    PROBE_TRAIN_SEED_BASE,
    SEED_STRIDE,
    interpret_probe_result,
    main,
    probe_seed_bases,
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


def test_main_reports_derivative_supervision_metadata(tmp_path):
    # Given: a tiny CPU derivative-supervised probe run.
    out_dir = tmp_path / "probe-derivaux"

    # When: the runner writes its report.
    main(
        out_dir=out_dir,
        n_steps=1,
        n_train_streams=2,
        n_test_streams=2,
        device="cpu",
        seed=0,
        seq_len=8,
        batch=1,
        d_model=16,
        n_layers=1,
        n_heads=4,
        lr=1e-3,
        d_conf_weight=10.0,
        dd_conf_weight=20.0,
    )

    # Then: report metadata makes the derivative-supervision condition explicit.
    report = json.loads((out_dir / "report.json").read_text())
    assert report["training"]["derivative_supervision_enabled"]
    assert report["training"]["d_conf_weight"] == 10.0
    assert report["training"]["dd_conf_weight"] == 20.0
    assert set(report["final_auxiliary_losses"]) == {
        "topmass_loss",
        "entropy_loss",
        "d_conf_loss",
        "dd_conf_loss",
    }
    assert report["run"]["n_steps"] == 1
    assert report["run"]["lr"] == 1e-3
    assert report["model"] == {"d_model": 16, "n_layers": 1, "n_heads": 4}
    assert report["stream"]["seq_len"] == 8


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
        probe_train_seed_base=10_000_000,
        probe_test_seed_base=20_000_000,
    )

    assert result.probe_train_seed_base == 10_000_000
    assert result.probe_test_seed_base == 20_000_000


def test_linear_probe_uses_closed_form_ridge_solution():
    # Given: a target that is exactly linear in hidden features.
    hidden = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    target = 2.0 * hidden[..., 0] - 3.0 * hidden[..., 1] + 0.5
    train_mask = torch.ones(target.shape, dtype=torch.bool)
    train_mask[0, 2] = False
    train_mask[1, 1] = False

    # When: the linear probe is fit without optimizer controls.
    result = linear_probe(hidden, target, "linear", train_mask)

    # Then: the closed-form fit succeeds without optimizer training.
    assert result.r2 > 0.99
    assert result.mse < 1e-4


def test_linear_probe_accepts_mps_device_request_without_mps_float64():
    # Given: a CPU tensor and an MPS device request from the default runner path.
    hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
    target = hidden[..., 0] + hidden[..., 1]
    train_mask = torch.tensor([[True, True], [False, False]])

    # When: a legacy device argument is passed to the closed-form probe.
    with pytest.warns(DeprecationWarning, match="closed-form CPU ridge"):
        result = linear_probe(hidden, target, "linear", train_mask, device="mps")

    # Then: solving happens on a supported dtype/device path.
    assert result.n_train == 2
    assert result.n_test == 2


def test_linear_probe_warns_for_positional_legacy_optimizer_args():
    # Given: a caller still using the old positional optimizer arguments.
    hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
    target = hidden[..., 0] + hidden[..., 1]
    train_mask = torch.tensor([[True, True], [False, False]])

    # When: deprecated positional arguments are provided.
    with pytest.warns(DeprecationWarning, match="n_epochs, lr"):
        result = linear_probe(hidden, target, "linear", train_mask, 300, 5e-3)

    # Then: the compatibility shim warns and still returns a closed-form fit.
    assert result.n_train == 2
    assert result.n_test == 2


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
