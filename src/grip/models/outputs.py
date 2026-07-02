from __future__ import annotations

from dataclasses import dataclass

import torch

from .sparse_components import SparseMetadata


@dataclass(frozen=True, slots=True)
class DenseModelOutput:
    lm_logits: torch.Tensor
    posterior: torch.Tensor
    hidden: torch.Tensor


@dataclass(frozen=True, slots=True)
class SparseModelOutput:
    lm_logits: torch.Tensor
    posterior: torch.Tensor
    hidden: torch.Tensor
    selected_blocks: torch.Tensor
    selection_scores: torch.Tensor
    metadata: SparseMetadata
    grip_state: torch.Tensor | None
    grip_recon: torch.Tensor | None


ModelOutput = DenseModelOutput | SparseModelOutput
