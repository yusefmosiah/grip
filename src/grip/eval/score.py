"""Score one or more runs and emit the comparison. Separate from training.

Usage: python -m grip.eval.score run_dir1 run_dir2 ...
Reads each run's eval tensors + config, computes all metrics, writes a
comparison JSON. This is where 'who won' is decided — never in the trainer.
"""
from __future__ import annotations
import json
from pathlib import Path


def score_run(run_dir: Path) -> dict:
    """Load artifacts from run_dir, compute the full metric set, return dict."""
    raise NotImplementedError("CODEX")


def compare(runs: list[Path]) -> dict:
    """Side-by-side table of all metrics across runs. Writes comparison.json."""
    raise NotImplementedError("CODEX")


if __name__ == "__main__":
    raise SystemExit("CODEX: wire CLI args -> compare([Path(a) for a in argv[1:]])")
