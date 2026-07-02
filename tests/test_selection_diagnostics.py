from __future__ import annotations

import torch

from grip.eval.selection_diagnostics import selection_diagnostics


def test_selection_diagnostics_uses_true_position_block_ids() -> None:
    # Given: selected block ids whose selected-rank order would give the wrong answer.
    selected_blocks = torch.tensor([[[0, 1], [0, 1], [2, 3], [1, 3]]])
    decisive_idx = torch.tensor([[0, 1, 1, 1]])

    # When: diagnostics derive true block ids from sequence position and block size.
    diagnostics = selection_diagnostics(
        selected_blocks=selected_blocks,
        decisive_idx=decisive_idx,
        attention_mode="content_sparse",
        block_size=2,
        read_budget=2,
    )

    # Then: recall uses position blocks [0, 0, 1, 1], not selected rank/order.
    assert diagnostics.decisive_token_count == 3
    assert abs(diagnostics.decisive_token_recall - (2.0 / 3.0)) < 1e-6
    assert diagnostics.selection_consumed is True


def test_selection_diagnostics_labels_local_selection_as_not_consumed() -> None:
    # Given: local mode still emits selected_blocks for trace comparability.
    selected_blocks = torch.tensor([[[0], [0], [1], [1]]])
    decisive_idx = torch.tensor([[0, 1, 0, 1]])

    # When: diagnostics are produced for the local baseline.
    diagnostics = selection_diagnostics(
        selected_blocks=selected_blocks,
        decisive_idx=decisive_idx,
        attention_mode="local",
        block_size=2,
        read_budget=1,
    )

    # Then: the report names the selector surface without claiming local consumed it.
    assert diagnostics.attention_mode == "local"
    assert diagnostics.selection_consumed is False


def test_selection_diagnostics_labels_grip_selection_as_consumed() -> None:
    # Given: a grip-select trace over selected blocks.
    selected_blocks = torch.tensor([[[0], [0], [1], [1]]])
    decisive_idx = torch.tensor([[0, 1, 0, 1]])

    # When: diagnostics are produced for grip-select mode.
    diagnostics = selection_diagnostics(
        selected_blocks=selected_blocks,
        decisive_idx=decisive_idx,
        attention_mode="grip_select",
        block_size=2,
        read_budget=1,
    )

    # Then: the report records that selected blocks are consumed by the model.
    assert diagnostics.attention_mode == "grip_select"
    assert diagnostics.selection_consumed is True
