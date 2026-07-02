from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from grip.models import ContentSparseTransformer, DenseTransformer
from grip.models.outputs import ModelOutput, SparseModelOutput

from .compute import compute_budget, compute_payload
from .experiment_config import experiment_provenance_payload
from .headroom_baselines import baseline_specs
from .headroom_training import BatchTensors, TrainingLoopConfig, TrainingTokenBatch, next_token_loss, train_model
from .headroom_types import BaselineSpec, HeadroomConfigError, MRegimeConfig, ResolvedJson
from .metrics import accuracy, brier_score, ece, mutual_info_discrete, nll
from .m_regime_validity import run_tier, run_validity
from .selection_diagnostics import write_selection_diagnostics


def write_baselines(
    config: MRegimeConfig,
    train_batches: tuple[TrainingTokenBatch, ...],
    eval_batch: BatchTensors,
) -> tuple[Path, ...]:
    return tuple(
        _write_baseline(config, spec, train_batches, eval_batch)
        for spec in baseline_specs(config.top_k_blocks)
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
    eval_real_mask = eval_batch["real_mask"]
    with torch.no_grad():
        out = model(eval_tokens, real_mask=eval_real_mask)
        loss = next_token_loss(
            logits=out.lm_logits,
            tokens=eval_tokens,
            real_mask=eval_real_mask,
            vocab_size=config.vocab_size,
        )
        metric_payload = _metric_payload(out, eval_batch, eval_real_mask, loss)
    compute = compute_budget(model, eval_tokens, read_budget=spec.read_budget)
    if spec.attention_mode is not None and spec.read_budget is not None:
        if not isinstance(out, SparseModelOutput):
            raise HeadroomConfigError("model_output", "sparse diagnostics require sparse output")
        write_selection_diagnostics(
            run_dir / "selection_diagnostics.json",
            selected_blocks=out.selected_blocks,
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
                "compute": compute_payload(compute),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metric_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in train_records) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _metric_payload(
    model_output: ModelOutput,
    eval_batch: BatchTensors,
    real_mask: torch.Tensor,
    loss: torch.Tensor,
) -> Mapping[str, float]:
    posterior = model_output.posterior
    last_real = real_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(posterior.shape[0], device=posterior.device)
    final_probs = posterior[batch_idx, last_real].clamp_min(1e-8)
    final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)
    answer = eval_batch["answer"].to(device=posterior.device, dtype=torch.long)
    source_idx = eval_batch["source_idx"].to(device=posterior.device, dtype=torch.long)
    expanded_answers = answer.reshape(-1, 1).expand_as(source_idx)
    real_source_idx = source_idx[real_mask]
    real_answers = expanded_answers[real_mask]
    return {
        "loss": float(loss.item()),
        "posterior_accuracy": accuracy(final_probs, answer),
        "posterior_brier": brier_score(final_probs, answer),
        "posterior_ece": ece(final_probs, answer),
        "posterior_nll": nll(final_probs.log(), answer),
        "source_answer_mi": mutual_info_discrete(real_source_idx, real_answers),
        "tokens": float(eval_batch["tokens"].numel()),
    }


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
    payload = experiment_provenance_payload(
        config,
        decision_seed_count=config.decision_seed_count,
        include_device=False,
    )
    model = payload["model"]
    if not isinstance(model, dict):
        raise HeadroomConfigError("model", "resolved payload model must be an object")
    model["attention_mode"] = spec.attention_mode
    model["name"] = spec.name
    eval_payload = payload["eval"]
    if not isinstance(eval_payload, dict):
        raise HeadroomConfigError("eval", "resolved payload eval must be an object")
    eval_payload["seed"] = config.seed + config.eval_seed_offset
    payload.update({
        "artifact_schema_version": 1,
        "read_budget": spec.read_budget,
        "run": {
            "device": config.device,
            "mode": "preregistered" if config.preregistered else "smoke",
        },
        "seed": config.seed,
    })
    validity_failures = run_validity(payload)
    tier = run_tier(payload)
    return {
        **payload,
        "tier": tier,
        "unciteable": tier != "valid",
        "validity_failures": list(validity_failures),
    }
