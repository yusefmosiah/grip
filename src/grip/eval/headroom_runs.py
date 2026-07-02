from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from grip.models import ContentSparseTransformer, DenseTransformer

from .headroom_training import BatchTensors, TrainingLoopConfig, TrainingTokenBatch, train_model
from .headroom_types import BaselineSpec, HeadroomConfigError, MRegimeConfig, ResolvedJson
from .m_regime_validity import run_tier, run_validity
from .selection_diagnostics import write_selection_diagnostics


def write_baselines(
    config: MRegimeConfig,
    train_batches: tuple[TrainingTokenBatch, ...],
    eval_batch: BatchTensors,
) -> tuple[Path, ...]:
    return tuple(
        _write_baseline(config, spec, train_batches, eval_batch)
        for spec in _baseline_specs(config)
    )


def _baseline_specs(config: MRegimeConfig) -> tuple[BaselineSpec, ...]:
    return (
        BaselineSpec("dense", None, None),
        BaselineSpec("local", "local", config.top_k_blocks),
        BaselineSpec("content-sparse", "content_sparse", config.top_k_blocks),
        BaselineSpec("grip-read-A", "grip_read", config.top_k_blocks),
        BaselineSpec("grip-select-B", "grip_select", config.top_k_blocks),
    )


def _write_baseline(
    config: MRegimeConfig,
    spec: BaselineSpec,
    train_batches: tuple[TrainingTokenBatch, ...],
    eval_batch: BatchTensors,
) -> Path:
    run_dir = config.out_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _build_seeded_model(config, spec)
    train_records = train_model(
        model=model,
        batches=train_batches,
        config=TrainingLoopConfig(
            dry_run_seed=config.seed,
            lr=config.lr,
            vocab_size=config.vocab_size,
        ),
    )
    model.eval()
    eval_tokens = eval_batch["tokens"]
    with torch.no_grad():
        out = model(eval_tokens)
        loss = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, config.vocab_size),
            eval_tokens[:, 1:].reshape(-1),
        )
    if spec.attention_mode is not None and spec.read_budget is not None:
        write_selection_diagnostics(
            run_dir / "selection_diagnostics.json",
            selected_blocks=out["selected_blocks"],
            decisive_idx=eval_batch["decisive_idx"],
            attention_mode=spec.attention_mode,
            block_size=config.block_size,
            read_budget=spec.read_budget,
        )
    eval_seed = config.seed + config.eval_seed_offset
    (run_dir / "config.resolved.json").write_text(
        json.dumps(_resolved_payload(config, spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_tensors.json").write_text(
        json.dumps(
            {
                "batch_size": config.eval_batch_size,
                "loss": float(loss.item()),
                "seed": eval_seed,
                "seed_offset": config.eval_seed_offset,
                "tokens": float(eval_tokens.numel()),
            },
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
    spec: BaselineSpec,
) -> DenseTransformer | ContentSparseTransformer:
    torch.manual_seed(config.seed)
    model = _build_model(config, spec)
    model.to(config.device)
    return model


def _build_model(
    config: MRegimeConfig,
    spec: BaselineSpec,
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


def _resolved_payload(config: MRegimeConfig, spec: BaselineSpec) -> Mapping[str, ResolvedJson]:
    payload = {
        "artifact_schema_version": 1,
        "data": {
            "seq_len": config.seq_len,
            "task": config.task,
            "vocab_size": config.vocab_size,
        },
        "decision": {
            "seed_count": config.decision_seed_count,
        },
        "model": {
            "attention_mode": spec.attention_mode,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_hypotheses": config.n_hypotheses,
            "n_layers": config.n_layers,
            "name": spec.name,
        },
        "read_budget": spec.read_budget,
        "run": {
            "device": config.device,
            "mode": "preregistered" if config.preregistered else "smoke",
        },
        "eval": {
            "batch_size": config.eval_batch_size,
            "seed": config.seed + config.eval_seed_offset,
            "seed_offset": config.eval_seed_offset,
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
    validity_failures = run_validity(payload)
    tier = run_tier(payload)
    return {
        **payload,
        "tier": tier,
        "unciteable": tier != "valid",
        "validity_failures": list(validity_failures),
    }
