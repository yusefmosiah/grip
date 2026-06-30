"""Minimal training loop.

The trainer's job: produce artifacts (checkpoint, JSONL log, eval tensors).
The trainer does NOT decide who won — that's eval/score.py's job.

Accepts a config dict (or YAML path). Configs are reproducibility: same config
+ same seed => same run (modulo MPS nondeterminism, which we log).
"""
from __future__ import annotations
from pathlib import Path
import json


def train(config: dict, run_dir: str | Path = "runs/default") -> Path:
    """Run one training job. Writes JSONL log + checkpoint into run_dir.

    config keys (minimal):
        model: {name: 'dense'|'sparse', d_model, n_layers, ...}
        data:  {task:'bayesian', seq_len, num_hypotheses, ...}
        train: {steps, microbatch, grad_accum, lr, precision, seed}
        device: 'mps' | 'cpu'
    """
    raise NotImplementedError(
        "CODEX: implement after dense model + streams + metrics land. "
        "Std loop: sample batch -> forward -> loss (LM + aux posterior) -> "
        "backward (grad-accum) -> step. Log step/tokens/loss/mem every N steps. "
        "Save checkpoint on a schedule + at end."
    )


if __name__ == "__main__":
    raise SystemExit(
        "Call via: python -m grip.train.run <config.yaml>. "
        "Config loader + CLI to be added by CODEX."
    )
