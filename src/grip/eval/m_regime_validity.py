from __future__ import annotations

from typing import Mapping

from .score_types import JsonValue


MIN_VALID_SEEDS = 8
MIN_VALID_SEQ_LEN = 512
MIN_VALID_EVAL_BATCH_SIZE = 8
MIN_VALID_TRAIN_BATCH_SIZE = 8
MIN_VALID_TRAIN_STEPS = 1000
MIN_VALID_CALIBRATION_PAIRS = 8


def run_validity(config: Mapping[str, JsonValue]) -> tuple[str, ...]:
    failures: list[str] = []
    if _get_int(config, ("decision", "seed_count")) < MIN_VALID_SEEDS:
        failures.append("decision.seed_count")
    if _get_int(config, ("data", "seq_len")) < MIN_VALID_SEQ_LEN:
        failures.append("data.seq_len")
    if _get_int(config, ("eval", "batch_size")) < MIN_VALID_EVAL_BATCH_SIZE:
        failures.append("eval.batch_size")
    if _get_int(config, ("train", "batch_size")) < MIN_VALID_TRAIN_BATCH_SIZE:
        failures.append("train.batch_size")
    if _get_int(config, ("train", "steps")) < MIN_VALID_TRAIN_STEPS:
        failures.append("train.steps")
    return tuple(failures)


def noise_floor_validity(calibration_pair_count: int) -> tuple[str, ...]:
    if calibration_pair_count < MIN_VALID_CALIBRATION_PAIRS:
        return ("noise_floor.calibration_pairs",)
    return ()


def run_tier(config: Mapping[str, JsonValue]) -> str:
    return "valid" if not run_validity(config) else "smoke"


def _get_int(config: Mapping[str, JsonValue], path: tuple[str, ...]) -> int:
    current: JsonValue = dict(config)
    for part in path:
        if not isinstance(current, dict):
            return -1
        current = current.get(part)
    if not isinstance(current, int) or isinstance(current, bool):
        return -1
    return current
