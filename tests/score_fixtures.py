from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from grip.eval.noise_floor_artifact import attach_noise_floor_content_hash


def write_run(
    run_dir: Path,
    metrics: Mapping[str, float],
    *,
    valid: bool = True,
    parameter_count: int = 1_000,
    estimated_forward_flops: int = 2_000,
    read_budget: int | None = None,
) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
    (run_dir / "config.resolved.json").write_text(
        json.dumps(run_config(run_dir.name, valid=valid), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_tensors.json").write_text(
        json.dumps(
            {
                "compute": {
                    "estimated_forward_flops": estimated_forward_flops,
                    "parameter_count": parameter_count,
                    "read_budget": read_budget,
                    "token_count": 512,
                },
                "loss": float(metrics.get("loss", 0.0)),
                "tokens": 512.0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def run_config(name: str, *, seq_len: int | None = None, valid: bool = True) -> dict:
    resolved_seq_len = seq_len if seq_len is not None else (512 if valid else 8)
    eval_batch_size = 8 if valid else 1
    train_batch_size = 8 if valid else 1
    train_steps = 1000 if valid else 0
    tier = "valid" if valid else "smoke"
    return {
        "decision": {"seed_count": 8 if valid else 1},
        "data": {"seq_len": resolved_seq_len, "task": "bayesian", "vocab_size": 17},
        "eval": {"batch_size": eval_batch_size, "seed": 10_000, "seed_offset": 10_000},
        "model": {
            "attention_mode": None,
            "d_model": 16,
            "n_heads": 4,
            "n_hypotheses": 3,
            "n_layers": 1,
            "name": name,
        },
        "read_budget": None,
        "run": {"device": "cpu", "mode": "preregistered"},
        "seed": 0,
        "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
        "tier": tier,
        "train": {"batch_size": train_batch_size, "lr": 1e-3, "steps": train_steps},
        "unciteable": not valid,
        "validity_failures": [] if valid else ["train.steps"],
    }


def write_noise_floor(path: Path, *, seed_count: int = 8) -> Path:
    accuracy_deltas = [0.01, -0.02, 0.0, 0.015, -0.01, 0.005, 0.02, -0.015][:seed_count]
    brier_deltas = [0.001, -0.002, 0.0, 0.0015, -0.001, 0.0005, 0.002, -0.0015][:seed_count]
    payload = {
        "kind": "M-noise-floor",
        "calibration": {
            "baseline_names": ["run-a", "run-b", "dense", "local", "content-sparse"],
            "decision": {"seed_count": 8},
            "data": {"seq_len": 512, "task": "bayesian", "vocab_size": 17},
            "device": "cpu",
            "eval": {"batch_size": 8, "seed_offset": 10_000},
            "model": {"d_model": 16, "n_heads": 4, "n_hypotheses": 3, "n_layers": 1},
            "sparse": {"block_size": 2, "top_k_blocks": 3, "window": 2},
            "train": {"batch_size": 8, "lr": 1e-3, "steps": 1000},
        },
        "seed_count": seed_count,
        "seed_ids": list(range(seed_count)),
        "calibration_pairs": [
            {
                "left": f"seed-{seed}-a",
                "left_seed": seed + 100,
                "right": f"seed-{seed}-b",
                "right_seed": seed + 200,
            }
            for seed in range(seed_count)
        ],
        "minimum_signal_threshold": {"accuracy": 0.02, "brier": 0.002},
        "metric_ceilings": {"accuracy": 0.02, "brier": 0.002},
        "metric_deltas": {
            "accuracy": accuracy_deltas,
            "brier": brier_deltas,
        },
        "zero_tolerance": 1e-12,
    }
    path.write_text(json.dumps(attach_noise_floor_content_hash(payload)), encoding="utf-8")
    return path
