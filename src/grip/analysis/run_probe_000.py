"""Minimal trainer for producing a backbone to probe.

Just enough to fit a DenseTransformer on the streams for the gating experiment.
This is NOT the full training loop (that's src/grip/train/run.py) — it's a
focused script: train a backbone with LM + aux-posterior loss, save it, then
the probe reads its hidden states.

Run: python -m grip.analysis.run_probe_000
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

from grip.data import BayesianEvidenceStream
from .probe import ProbeExperimentResult, run_probe_experiment
from .probe_training import (
    BACKBONE_BATCH_SEED_STRIDE,
    BACKBONE_SEED_LIMIT,
    SupervisionWeights,
    backbone_batch_seed_base,
    train_backbone,
    train_backbone_with_report,
)


LEVEL_CONTROL_R2_THRESHOLD: Final = 0.5
DERIVATIVE_R2_THRESHOLD: Final = 0.2
PROBE_TRAIN_SEED_BASE: Final = 10_000_000_000_000
PROBE_TEST_SEED_BASE: Final = 20_000_000_000_000
SEED_STRIDE: Final = 1_000_000
PROBE_SEED_LIMIT: Final = (PROBE_TEST_SEED_BASE - PROBE_TRAIN_SEED_BASE) // SEED_STRIDE
__all__: Final = ("main", "probe_seed_bases", "train_backbone")


@dataclass(frozen=True, slots=True)
class ProbeInterpretation:
    status: str
    level_control_passed: bool
    message: str


def probe_seed_bases(
    seed: int,
    *,
    n_train_streams: int = 200,
    n_test_streams: int = 80,
) -> tuple[int, int]:
    seed_limit = min(PROBE_SEED_LIMIT, BACKBONE_SEED_LIMIT)
    if seed < 0 or seed >= seed_limit:
        msg = f"seed must be in [0, {seed_limit}) for disjoint probe/backbone namespaces"
        raise ValueError(msg)
    if n_train_streams < 0 or n_test_streams < 0:
        msg = "probe stream counts must be non-negative"
        raise ValueError(msg)
    if max(n_train_streams, n_test_streams) >= SEED_STRIDE:
        msg = "probe stream counts must fit within the per-seed stride"
        raise ValueError(msg)
    return (
        PROBE_TRAIN_SEED_BASE + seed * SEED_STRIDE,
        PROBE_TEST_SEED_BASE + seed * SEED_STRIDE,
    )


def interpret_probe_result(
    res: ProbeExperimentResult,
    level_threshold: float = LEVEL_CONTROL_R2_THRESHOLD,
    derivative_threshold: float = DERIVATIVE_R2_THRESHOLD,
) -> ProbeInterpretation:
    failed_controls = [
        name for name, result in res.level.items()
        if result.r2 < level_threshold
    ]
    if failed_controls:
        names = ", ".join(failed_controls)
        return ProbeInterpretation(
            status="invalid_level_control",
            level_control_passed=False,
            message=(
                f"Do not interpret derivative probes: level controls below "
                f"R^2 {level_threshold:.2f}: {names}."
            ),
        )

    readable = [
        name for name, result in res.derivative.items()
        if result.r2 >= derivative_threshold
    ]
    if not readable:
        return ProbeInterpretation(
            status="amnesia_supported",
            level_control_passed=True,
            message="Level controls passed and derivative probes are near baseline.",
        )
    if len(readable) == len(res.derivative):
        return ProbeInterpretation(
            status="trajectory_present",
            level_control_passed=True,
            message="Level controls passed and all derivative probes are readable.",
        )
    names = ", ".join(readable)
    return ProbeInterpretation(
        status="mixed_derivative_result",
        level_control_passed=True,
        message=f"Level controls passed; readable derivative probes: {names}.",
    )


def main(
    out_dir="runs/probe-000",
    n_steps=1500, n_train_streams=200, n_test_streams=80,
    device="mps", seed=0, seq_len=128, batch=8,
    d_model=128, n_layers=4, n_heads=4, lr=5e-4,
    lm_weight=0.1, aux_weight=5.0, topmass_weight=10.0, entropy_weight=3.0,
    d_conf_weight=0.0, dd_conf_weight=0.0,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stream = BayesianEvidenceStream(
        num_hypotheses=4, num_sources=3, seq_len=seq_len, vocab_size=64, seed=seed,
    )
    supervision = SupervisionWeights(
        lm=lm_weight,
        posterior=aux_weight,
        topmass=topmass_weight,
        entropy=entropy_weight,
        d_conf=d_conf_weight,
        dd_conf=dd_conf_weight,
    )
    print(f"=== training backbone on {device} ===")
    training_result = train_backbone_with_report(
        stream,
        n_steps=n_steps,
        device=device,
        seed=seed,
        batch=batch,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        lr=lr,
        supervision=supervision,
    )
    for p in training_result.model.parameters():
        p.requires_grad_(False)  # freeze
    print("\n=== running probe experiment (gating experiment #0) ===")
    if n_steps >= BACKBONE_BATCH_SEED_STRIDE:
        msg = "n_steps must fit within the per-backbone seed stride"
        raise ValueError(msg)
    probe_train_seed_base, probe_test_seed_base = probe_seed_bases(
        seed,
        n_train_streams=n_train_streams,
        n_test_streams=n_test_streams,
    )
    res = run_probe_experiment(
        training_result.model, stream, n_train_streams=n_train_streams,
        n_test_streams=n_test_streams, seed=seed, device=device,
        probe_train_seed_base=probe_train_seed_base,
        probe_test_seed_base=probe_test_seed_base,
    )

    def fmt(d): return {k: {"r2": round(v.r2, 4), "mse": round(v.mse, 6)} for k, v in d.items()}
    interpretation = interpret_probe_result(res)
    report = {
        "backbone_params": res.backbone_params,
        "n_train_streams": res.n_train_streams, "n_test_streams": res.n_test_streams,
        "probe_train_seed_base": res.probe_train_seed_base,
        "probe_test_seed_base": res.probe_test_seed_base,
        "level_control": fmt(res.level),
        "DERIVATIVE_TEST": fmt(res.derivative),
        "routing": fmt(res.routing),
        "interpretation": asdict(interpretation),
        "training": supervision.as_report(),
        "final_auxiliary_losses": training_result.final_losses.as_report(),
        "stream": {
            "num_hypotheses": stream.K,
            "num_sources": stream.S,
            "seq_len": stream.T,
            "vocab_size": stream.vocab_size,
        },
        "model": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
        },
        "run": {
            "seed": seed,
            "backbone_batch_seed_base": backbone_batch_seed_base(seed),
            "n_steps": n_steps,
            "batch": batch,
            "lr": lr,
            "device": device,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("GATING EXPERIMENT #0 — DERIVATIVE PROBE RESULT")
    print("=" * 60)
    print(f"\nbackbone: {res.backbone_params:,} params | "
          f"train streams={res.n_train_streams} test streams={res.n_test_streams}")
    print("\nLEVEL (control — expected to succeed):")
    for k, v in res.level.items():
        print(f"  {k:12s}  R^2 = {v.r2:+.4f}")
    print("\nDERIVATIVE (THE TEST):")
    for k, v in res.derivative.items():
        print(f"  {k:12s}  R^2 = {v.r2:+.4f}")
    print("\nROUTING:")
    for k, v in res.routing.items():
        print(f"  {k:18s}  R^2 = {v.r2:+.4f}")
    print("\n" + "=" * 60)
    print("INTERPRETATION (per SPEC-000 decision rule):")
    print(f"  status: {interpretation.status}")
    print(f"  {interpretation.message}")
    print("=" * 60)
    print(f"\nreport written to {out/'report.json'}")


if __name__ == "__main__":
    main()
