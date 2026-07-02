from __future__ import annotations

from typing import Sequence

from .noise_floor_calibration_types import NoiseFloorCalibrationConfig
from .score_types import JsonValue


BASELINE_NAMES: tuple[str, ...] = ("dense", "local", "content-sparse")


def calibration_payload(config: NoiseFloorCalibrationConfig) -> dict[str, JsonValue]:
    return {
        "baseline_names": list(BASELINE_NAMES),
        "data": {
            "seq_len": config.seq_len,
            "task": config.task,
            "vocab_size": config.vocab_size,
        },
        "decision": {
            "seed_count": len(config.decision_seed_ids),
        },
        "device": config.device,
        "eval": {
            "batch_size": config.eval_batch_size,
            "seed_offset": config.eval_seed_offset,
        },
        "model": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_hypotheses": config.n_hypotheses,
            "n_layers": config.n_layers,
        },
        "sparse": {
            "block_size": config.block_size,
            "top_k_blocks": config.top_k_blocks,
            "window": config.window,
        },
        "train": {
            "batch_size": config.train_batch_size,
            "lr": config.lr,
            "steps": config.train_steps,
        },
    }


def noise_floor_payload(
    config: NoiseFloorCalibrationConfig,
    pairs: Sequence[dict[str, JsonValue]],
    metric_deltas: dict[str, tuple[float, ...]],
    metric_ceilings: dict[str, float],
    minimum_signal_threshold: dict[str, float],
) -> dict[str, JsonValue]:
    return {
        "calibration": calibration_payload(config),
        "calibration_pairs": list(pairs),
        "kind": "M-noise-floor",
        "metric_ceilings": metric_ceilings,
        "metric_deltas": {
            metric: list(deltas)
            for metric, deltas in metric_deltas.items()
        },
        "minimum_signal_threshold": minimum_signal_threshold,
        "seed_count": len(config.seed_ids),
        "seed_ids": list(config.seed_ids),
        "zero_tolerance": config.minimum_signal_floor,
    }
