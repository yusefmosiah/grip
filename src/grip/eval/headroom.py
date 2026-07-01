from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from grip.models import ContentSparseTransformer, DenseTransformer

from .headroom_training import train_model, training_tokens
from .headroom_types import (
    BaselineSpec as _BaselineSpec,
    HeadroomConfigError,
    HeadroomStatus,
    MRegimeConfig,
    MRegimeResult,
    ResolvedJson,
)
from .score import compare
from .score_types import ComparisonReport


def run_m_regime_smoke(config: MRegimeConfig) -> MRegimeResult:
    _validate_config(config)
    torch.manual_seed(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = training_tokens(
        task=config.task,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_hypotheses=config.n_hypotheses,
        batch_size=config.train_batch_size,
        seed=config.seed,
        device=config.device,
    )
    eval_tokens = train_tokens[:1]
    run_dirs = tuple(
        _write_baseline(config, spec, train_tokens, eval_tokens)
        for spec in _baseline_specs(config)
    )
    comparison = compare(
        run_dirs,
        noise_floor_path=config.noise_floor_path,
        preregistered=config.preregistered,
    )
    status = _headroom_status(comparison)
    report_path = config.out_dir / "m_regime_report.json"
    report_path.write_text(
        json.dumps(
            {
                "authorize_avsb": status == "keep",
                "comparison_path": str(config.out_dir / "comparison.json"),
                "comparison_reason": comparison.reason,
                "interpretable": comparison.interpretable,
                "run_dirs": [str(path) for path in run_dirs],
                "status": status,
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


def _baseline_specs(config: MRegimeConfig) -> tuple[_BaselineSpec, ...]:
    return (
        _BaselineSpec("dense", None, None),
        _BaselineSpec("local", "local", config.top_k_blocks),
        _BaselineSpec("content-sparse", "content_sparse", config.top_k_blocks),
    )


def _write_baseline(
    config: MRegimeConfig,
    spec: _BaselineSpec,
    train_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
) -> Path:
    run_dir = config.out_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _build_seeded_model(config, spec)
    train_records = train_model(
        model=model,
        tokens=train_tokens,
        steps=config.train_steps,
        lr=config.lr,
        vocab_size=config.vocab_size,
    )
    model.eval()
    with torch.no_grad():
        out = model(eval_tokens)
        loss = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, config.vocab_size),
            eval_tokens[:, 1:].reshape(-1),
        )
    (run_dir / "config.resolved.json").write_text(
        json.dumps(_resolved_payload(config, spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_tensors.json").write_text(
        json.dumps(
            {"loss": float(loss.item()), "tokens": float(eval_tokens.numel())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in train_records) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _build_seeded_model(
    config: MRegimeConfig,
    spec: _BaselineSpec,
) -> DenseTransformer | ContentSparseTransformer:
    torch.manual_seed(config.seed)
    model = _build_model(config, spec)
    model.to(config.device)
    return model


def _build_model(
    config: MRegimeConfig,
    spec: _BaselineSpec,
) -> DenseTransformer | ContentSparseTransformer:
    if spec.name == "dense":
        return DenseTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.seq_len,
            n_hypotheses=config.n_hypotheses,
        )
    if spec.attention_mode is None:
        raise HeadroomConfigError("attention_mode", "sparse baseline requires attention_mode")
    return ContentSparseTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.seq_len,
        n_hypotheses=config.n_hypotheses,
        block_size=config.block_size,
        top_k_blocks=config.top_k_blocks,
        window=config.window,
        attention_mode=spec.attention_mode,
    )


def _resolved_payload(config: MRegimeConfig, spec: _BaselineSpec) -> Mapping[str, ResolvedJson]:
    return {
        "artifact_schema_version": 1,
        "data": {
            "seq_len": config.seq_len,
            "task": config.task,
            "vocab_size": config.vocab_size,
        },
        "model": {
            "attention_mode": spec.attention_mode,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "name": spec.name,
        },
        "read_budget": spec.read_budget,
        "run": {
            "device": config.device,
            "mode": "preregistered" if config.preregistered else "smoke",
        },
        "seed": config.seed,
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


def _validate_config(config: MRegimeConfig) -> None:
    if config.train_steps < 0:
        raise HeadroomConfigError("train_steps", "must be non-negative")
    if config.train_batch_size <= 0:
        raise HeadroomConfigError("train_batch_size", "must be positive")
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
