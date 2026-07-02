from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .metrics import decisive_token_recall


SelectionDiagnosticJson = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class SelectionDiagnosticsError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class SelectionDiagnostics:
    attention_mode: str
    block_size: int
    decisive_token_count: int
    decisive_token_recall: float | None
    read_budget: int
    selection_consumed: bool

    def as_json(self) -> dict[str, SelectionDiagnosticJson]:
        return {
            "attention_mode": self.attention_mode,
            "block_size": self.block_size,
            "decisive_token_count": self.decisive_token_count,
            "decisive_token_recall": self.decisive_token_recall,
            "read_budget": self.read_budget,
            "selection_consumed": self.selection_consumed,
        }


def selection_diagnostics(
    *,
    selected_blocks: torch.Tensor,
    decisive_idx: torch.Tensor,
    attention_mode: str,
    block_size: int,
    read_budget: int,
) -> SelectionDiagnostics:
    if block_size <= 0:
        raise SelectionDiagnosticsError("block_size", "must be positive")
    if read_budget <= 0:
        raise SelectionDiagnosticsError("read_budget", "must be positive")
    position_block_ids = torch.arange(decisive_idx.shape[1], device=selected_blocks.device) // block_size
    decisive_on_device = decisive_idx.to(device=selected_blocks.device)
    return SelectionDiagnostics(
        attention_mode=attention_mode,
        block_size=block_size,
        decisive_token_count=int(decisive_idx.bool().sum().item()),
        decisive_token_recall=decisive_token_recall(
            selected_blocks,
            decisive_on_device,
            position_block_ids,
        ),
        read_budget=read_budget,
        selection_consumed=_selection_consumed(attention_mode),
    )


def write_selection_diagnostics(
    path: Path,
    *,
    selected_blocks: torch.Tensor,
    decisive_idx: torch.Tensor,
    attention_mode: str,
    block_size: int,
    read_budget: int,
) -> SelectionDiagnostics:
    diagnostics = selection_diagnostics(
        selected_blocks=selected_blocks,
        decisive_idx=decisive_idx,
        attention_mode=attention_mode,
        block_size=block_size,
        read_budget=read_budget,
    )
    path.write_text(json.dumps(diagnostics.as_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return diagnostics


def _selection_consumed(attention_mode: str) -> bool:
    match attention_mode:
        case "content_sparse" | "grip_read" | "grip_select":
            return True
        case "local":
            return False
        case _:
            raise SelectionDiagnosticsError("attention_mode", "must be local, content_sparse, grip_read, or grip_select")
