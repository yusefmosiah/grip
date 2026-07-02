from __future__ import annotations

import json

import torch

from .headroom_runs import write_baselines
from .headroom_training import TrainingBatchRequest, training_batch, training_token_batches
from .headroom_types import (
    HeadroomConfigError,
    HeadroomStatus,
    MRegimeConfig,
    MRegimeResult,
)
from .m_regime_validity import run_tier, run_validity
from .score import compare
from .score_types import ComparisonReport


def run_m_regime_smoke(config: MRegimeConfig) -> MRegimeResult:
    _validate_config(config)
    torch.manual_seed(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    train_batches = training_token_batches(
        TrainingBatchRequest(
            task=config.task,
            seq_len=config.seq_len,
            vocab_size=config.vocab_size,
            n_hypotheses=config.n_hypotheses,
            batch_size=config.train_batch_size,
            seed=config.seed,
            steps=config.train_steps,
            device=config.device,
        )
    )
    eval_seed = config.seed + config.eval_seed_offset
    eval_batch = training_batch(
        task=config.task,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_hypotheses=config.n_hypotheses,
        batch_size=config.eval_batch_size,
        seed=eval_seed,
        device=config.device,
    )
    run_dirs = write_baselines(config, train_batches, eval_batch)
    comparison = compare(
        run_dirs,
        config.out_dir / "comparison.json",
        noise_floor_path=config.noise_floor_path,
        preregistered=config.preregistered,
    )
    status = _headroom_status(comparison)
    report_path = config.out_dir / "m_regime_report.json"
    tier = _m_regime_tier(config)
    validity_failures = _m_regime_validity_failures(config)
    report_path.write_text(
        json.dumps(
            {
                "authorize_avsb": status == "keep",
                "comparison_path": str(config.out_dir / "comparison.json"),
                "comparison_reason": comparison.reason,
                "interpretable": comparison.interpretable,
                "run_dirs": [str(path) for path in run_dirs],
                "status": status,
                "tier": tier,
                "unciteable": tier != "valid",
                "validity_failures": list(validity_failures),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return MRegimeResult(
        run_dirs=run_dirs,
        comparison=comparison,
        report_path=report_path,
        status=status,
        authorize_avsb=status == "keep",
    )


def _headroom_status(comparison: ComparisonReport) -> HeadroomStatus:
    if not comparison.interpretable:
        return "blocked"
    by_name = {score.run_dir.name: score for score in comparison.runs}
    dense_loss = by_name["dense"].metrics["loss"]
    content_sparse_loss = by_name["content-sparse"].metrics["loss"]
    noise_floor = comparison.noise_floor
    if noise_floor is None:
        return "blocked"
    threshold = noise_floor.minimum_signal_threshold["loss"]
    if content_sparse_loss - dense_loss > threshold:
        return "keep"
    return "pivot"


def _m_regime_tier(config: MRegimeConfig) -> str:
    return run_tier(_m_regime_validity_payload(config))


def _m_regime_validity_failures(config: MRegimeConfig) -> tuple[str, ...]:
    return run_validity(_m_regime_validity_payload(config))


def _m_regime_validity_payload(config: MRegimeConfig) -> dict:
    return {
        "data": {"seq_len": config.seq_len},
        "decision": {"seed_count": config.decision_seed_count},
        "eval": {"batch_size": config.eval_batch_size},
        "train": {
            "batch_size": config.train_batch_size,
            "steps": config.train_steps,
        },
    }


def _validate_config(config: MRegimeConfig) -> None:
    if config.train_steps < 0:
        raise HeadroomConfigError("train_steps", "must be non-negative")
    if config.train_batch_size <= 0:
        raise HeadroomConfigError("train_batch_size", "must be positive")
    if config.eval_batch_size <= 0:
        raise HeadroomConfigError("eval_batch_size", "must be positive")
    if config.eval_seed_offset <= 0:
        raise HeadroomConfigError("eval_seed_offset", "must be positive")
    if config.eval_seed_offset <= config.train_steps:
        raise HeadroomConfigError("eval_seed_offset", "must exceed train_steps")
    if config.lr <= 0:
        raise HeadroomConfigError("lr", "must be positive")
    if config.device != "cpu":
        raise HeadroomConfigError("device", "must be cpu for the minimal headroom gate")
    if config.task not in {"bayesian", "reversal"}:
        raise HeadroomConfigError("task", "must be bayesian or reversal")
    if config.task == "reversal" and config.seq_len < 16:
        raise HeadroomConfigError("seq_len", "must be >= 16 for reversal")
    if config.task == "reversal" and config.vocab_size != 64:
        raise HeadroomConfigError("vocab_size", "must be 64 for reversal")
    if config.task == "reversal" and config.n_hypotheses != 4:
        raise HeadroomConfigError("n_hypotheses", "must be 4 for reversal")
