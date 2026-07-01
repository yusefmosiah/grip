from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

import torch
import torch.nn.functional as F

from grip.models import ContentSparseTransformer, DenseTransformer

from .score import compare
from .score_types import ComparisonReport


HeadroomStatus: TypeAlias = Literal["keep", "pivot", "blocked"]
ResolvedJson: TypeAlias = str | int | bool | None | Mapping[str, str | int | None]


@dataclass(frozen=True, slots=True)
class HeadroomConfigError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class MRegimeConfig:
    out_dir: Path
    noise_floor_path: Path | None = None
    preregistered: bool = False
    seed: int = 0
    seq_len: int = 8
    vocab_size: int = 17
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 1
    n_hypotheses: int = 3
    block_size: int = 2
    top_k_blocks: int = 3
    window: int = 2


@dataclass(frozen=True, slots=True)
class MRegimeResult:
    run_dirs: tuple[Path, ...]
    comparison: ComparisonReport
    report_path: Path
    status: HeadroomStatus
    authorize_avsb: bool


@dataclass(frozen=True, slots=True)
class _BaselineSpec:
    name: str
    attention_mode: str | None
    read_budget: int | None


def run_m_regime_smoke(config: MRegimeConfig) -> MRegimeResult:
    torch.manual_seed(config.seed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    tokens = torch.arange(config.seq_len, dtype=torch.long).remainder(config.vocab_size).unsqueeze(0)
    run_dirs = tuple(_write_baseline(config, spec, tokens) for spec in _baseline_specs(config))
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
    tokens: torch.Tensor,
) -> Path:
    run_dir = config.out_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(config, spec).eval()
    with torch.no_grad():
        out = model(tokens)
        loss = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
    (run_dir / "config.resolved.json").write_text(
        json.dumps(_resolved_payload(config, spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_tensors.json").write_text(
        json.dumps(
            {"loss": float(loss.item()), "tokens": float(tokens.numel())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


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
            "task": "m-regime-smoke",
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
        "run": {"mode": "preregistered" if config.preregistered else "smoke"},
        "seed": config.seed,
        "sparse": {
            "block_size": config.block_size,
            "top_k_blocks": config.top_k_blocks,
            "window": config.window,
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
