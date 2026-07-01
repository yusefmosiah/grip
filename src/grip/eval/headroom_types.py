from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

from .score_types import ComparisonReport


HeadroomStatus: TypeAlias = Literal["keep", "pivot", "blocked"]
ResolvedJson: TypeAlias = str | int | float | bool | None | Mapping[str, str | int | float | None]


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
    task: str = "bayesian"
    device: str = "cpu"
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
    train_steps: int = 0
    train_batch_size: int = 1
    lr: float = 1e-3


@dataclass(frozen=True, slots=True)
class MRegimeResult:
    run_dirs: tuple[Path, ...]
    comparison: ComparisonReport
    report_path: Path
    status: HeadroomStatus
    authorize_avsb: bool


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    name: str
    attention_mode: str | None
    read_budget: int | None
