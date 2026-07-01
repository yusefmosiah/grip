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
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from grip.data import BayesianEvidenceStream, make_batch
from grip.models import DenseTransformer
from .probe import ProbeExperimentResult, run_probe_experiment


LEVEL_CONTROL_R2_THRESHOLD = 0.5
DERIVATIVE_R2_THRESHOLD = 0.2
PROBE_TRAIN_SEED_BASE = 10_000_000
PROBE_TEST_SEED_BASE = 20_000_000
SEED_STRIDE = 1_000_000


@dataclass(frozen=True, slots=True)
class ProbeInterpretation:
    status: str
    level_control_passed: bool
    message: str


def probe_seed_bases(seed: int) -> tuple[int, int]:
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


def train_backbone(
    stream, d_model=128, n_layers=4, n_heads=4,
    n_steps=1500, batch=8, lr=5e-4, device="mps", seed=0,
    lm_weight=0.1, aux_weight=5.0, topmass_weight=10.0,
    entropy_weight=3.0, log_every=100,
):
    torch.manual_seed(seed)
    model = DenseTransformer(
        vocab_size=stream.vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_seq_len=stream.T, n_hypotheses=stream.K,
    ).to(device)
    topmass_head = nn.Linear(d_model, 1).to(device)
    entropy_head = nn.Linear(d_model, 1).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(topmass_head.parameters()) + list(entropy_head.parameters()),
        lr=lr,
    )
    seed_seq = seed * 13 + 1
    t0 = time.time()
    for step in range(n_steps):
        # fresh batches each step (online generation; streams are cheap)
        batch_data = make_batch(stream, n=batch, seed=seed_seq + step, device=device)
        out = model(batch_data["tokens"])
        hidden = out["hidden"]
        lm = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, stream.vocab_size),
            batch_data["tokens"][:, 1:].reshape(-1),
        )
        log_p = torch.log(out["posterior"] + 1e-8)
        aux = (batch_data["posterior"] * (
            torch.log(batch_data["posterior"] + 1e-8) - log_p)).sum(-1).mean()
        topmass = batch_data["posterior"].max(-1).values
        topmass_loss = F.mse_loss(topmass_head(hidden).squeeze(-1), topmass)
        entropy_loss = F.mse_loss(entropy_head(hidden).squeeze(-1), batch_data["entropy"])
        loss = (
            lm_weight * lm
            + aux_weight * aux
            + topmass_weight * topmass_loss
            + entropy_weight * entropy_loss
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            mem = (torch.mps.current_allocated_memory() / 1e6
                   if device == "mps" and hasattr(torch.mps, "current_allocated_memory") else 0)
            print(f"  step {step:4d}  lm={lm.item():.3f}  aux={aux.item():.4f}  "
                  f"top={topmass_loss.item():.4f}  ent={entropy_loss.item():.4f}  "
                  f"loss={loss.item():.3f}  mem={mem:.0f}MB  dt={time.time()-t0:.0f}s")
    return model


def main(
    out_dir="runs/probe-000",
    n_steps=1500, n_train_streams=200, n_test_streams=80,
    device="mps", seed=0, seq_len=128, batch=8,
    lm_weight=0.1, aux_weight=5.0, topmass_weight=10.0, entropy_weight=3.0,
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stream = BayesianEvidenceStream(
        num_hypotheses=4, num_sources=3, seq_len=seq_len, vocab_size=64, seed=seed,
    )
    print(f"=== training backbone on {device} ===")
    model = train_backbone(stream, n_steps=n_steps, device=device, seed=seed,
                           batch=batch, lm_weight=lm_weight, aux_weight=aux_weight,
                           topmass_weight=topmass_weight, entropy_weight=entropy_weight)
    for p in model.parameters():
        p.requires_grad_(False)  # freeze
    print(f"\n=== running probe experiment (gating experiment #0) ===")
    probe_train_seed_base, probe_test_seed_base = probe_seed_bases(seed)
    res = run_probe_experiment(
        model, stream, n_train_streams=n_train_streams,
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
        "training": {
            "lm_weight": lm_weight,
            "aux_weight": aux_weight,
            "topmass_weight": topmass_weight,
            "entropy_weight": entropy_weight,
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
