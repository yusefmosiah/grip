from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .noise_floor import MIN_NOISE_FLOOR_SEEDS
from .noise_floor_calibration import calibrate_noise_floor
from .noise_floor_calibration_types import (
    DEFAULT_CALIBRATION_SEED_OFFSET,
    NoiseFloorCalibrationConfig,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--task", choices=("bayesian", "reversal"), default="bayesian")
    parser.add_argument("--seed-count", type=int, default=MIN_NOISE_FLOOR_SEEDS)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-seed-offset", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=17)
    parser.add_argument("--n-hypotheses", type=int, default=3)
    args = parser.parse_args(argv)
    result = calibrate_noise_floor(
        NoiseFloorCalibrationConfig(
            out_dir=args.out_dir,
            task=args.task,
            seed_ids=tuple(
                DEFAULT_CALIBRATION_SEED_OFFSET + seed
                for seed in range(args.seed_count)
            ),
            decision_seed_ids=tuple(range(args.seed_count)),
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            n_hypotheses=args.n_hypotheses,
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_seed_offset=args.eval_seed_offset,
        )
    )
    print(result.path)
    return 0
