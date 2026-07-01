import pytest
import torch

from grip.models import ContentSparseTransformer


def _tiny_sparse_model(attention_mode: str = "content_sparse") -> ContentSparseTransformer:
    return ContentSparseTransformer(
        vocab_size=17,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=8,
        n_hypotheses=3,
        block_size=2,
        top_k_blocks=3,
        window=2,
        attention_mode=attention_mode,
    )


def test_sparse_forward_returns_trace_contract() -> None:
    # Given: a tiny sparse model and a fixed token batch.
    model = _tiny_sparse_model().eval()
    tokens = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]],
        dtype=torch.long,
    )

    # When: the model runs a forward pass.
    with torch.no_grad():
        out = model(tokens)

    # Then: every sparse/grip trace field has the agreed shape and placeholder state.
    assert set(out) == {
        "lm_logits",
        "posterior",
        "hidden",
        "selected_blocks",
        "selection_scores",
        "metadata",
        "grip_state",
        "grip_recon",
    }
    assert out["lm_logits"].shape == (2, 8, 17)
    assert out["posterior"].shape == (2, 8, 3)
    assert out["hidden"].shape == (2, 8, 16)
    assert out["selected_blocks"].shape == (2, 8, 3)
    assert out["selection_scores"].shape == (2, 8, 4)
    assert out["metadata"] == {
        "attention_mode": "content_sparse",
        "block_size": 2,
        "read_budget": 3,
        "window": 2,
    }
    assert out["grip_state"] is None
    assert out["grip_recon"] is None
    assert torch.allclose(out["posterior"].sum(-1), torch.ones(2, 8), atol=1e-6)


def test_sparse_selection_is_causal_at_block_boundaries() -> None:
    # Given: four two-token blocks and a model with a three-block read budget.
    model = _tiny_sparse_model().eval()
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: the model emits sparse selection traces.
    with torch.no_grad():
        out = model(tokens)

    # Then: selected block ids and finite scores never point into future blocks.
    current_block = torch.arange(8) // 2
    selected_blocks = out["selected_blocks"][0]
    selection_scores = out["selection_scores"][0]
    assert torch.all(selected_blocks <= current_block[:, None])
    for token_idx, block_idx in enumerate(current_block.tolist()):
        future_scores = selection_scores[token_idx, block_idx + 1 :]
        assert torch.isneginf(future_scores).all()


def test_sparse_selection_scores_are_prefix_causal_within_current_block() -> None:
    # Given: two batches that differ only at a future token in token 0's block.
    torch.manual_seed(123)
    model = _tiny_sparse_model().eval()
    original = torch.arange(8, dtype=torch.long).unsqueeze(0)
    changed = original.clone()
    changed[0, 1] = 16

    # When: the model emits selector scores for both batches.
    with torch.no_grad():
        original_out = model(original)
        changed_out = model(changed)

    # Then: token 0's hidden state and selector scores are unchanged.
    assert torch.equal(original_out["hidden"][:, 0], changed_out["hidden"][:, 0])
    assert torch.equal(
        original_out["selection_scores"][:, 0],
        changed_out["selection_scores"][:, 0],
    )


def test_sparse_forward_is_deterministic_in_eval_mode() -> None:
    # Given: one initialized model in eval mode and a repeated input batch.
    torch.manual_seed(123)
    model = _tiny_sparse_model().eval()
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: the same model evaluates the same tokens twice.
    with torch.no_grad():
        first = model(tokens)
        second = model(tokens)

    # Then: all tensor-valued trace fields are exactly repeatable.
    for key in ("lm_logits", "posterior", "hidden", "selected_blocks", "selection_scores"):
        assert torch.equal(first[key], second[key])


def test_sparse_modes_report_metadata_and_content_sparse_consumes_context() -> None:
    # Given: local-only and content-sparse models with identical weights.
    torch.manual_seed(7)
    local_model = _tiny_sparse_model(attention_mode="local").eval()
    content_model = _tiny_sparse_model(attention_mode="content_sparse").eval()
    content_model.load_state_dict(local_model.state_dict())
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: both modes evaluate the same sequence.
    with torch.no_grad():
        local_out = local_model(tokens)
        content_out = content_model(tokens)

    # Then: metadata identifies the mode and content-sparse consumes selected context.
    assert local_out["metadata"]["attention_mode"] == "local"
    assert content_out["metadata"]["attention_mode"] == "content_sparse"
    assert not torch.equal(local_out["hidden"], content_out["hidden"])
    assert torch.equal(local_out["selected_blocks"], content_out["selected_blocks"])


def test_content_sparse_hidden_depends_on_selected_block_ids() -> None:
    class PreferBlock(ContentSparseTransformer):
        def __init__(self, preferred_block: int):
            super().__init__(
                vocab_size=17,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_seq_len=8,
                n_hypotheses=3,
                block_size=2,
                top_k_blocks=1,
                window=2,
            )
            self.preferred_block = preferred_block

        def _block_importance(
            self,
            query: torch.Tensor,
            block_summaries: torch.Tensor,
        ) -> torch.Tensor:
            scores = torch.zeros(
                query.shape[0],
                query.shape[1],
                block_summaries.shape[2],
                device=query.device,
                dtype=query.dtype,
            )
            scores[..., self.preferred_block] = 1
            return scores

    # Given: two content-sparse models with identical weights and different selectors.
    torch.manual_seed(11)
    prefer_oldest = PreferBlock(preferred_block=0).eval()
    prefer_current = PreferBlock(preferred_block=3).eval()
    prefer_current.load_state_dict(prefer_oldest.state_dict())
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: both models evaluate a token whose past and current blocks are visible.
    with torch.no_grad():
        oldest_out = prefer_oldest(tokens)
        current_out = prefer_current(tokens)

    # Then: changing only selected block ids changes the consumed-context hidden state.
    assert oldest_out["selected_blocks"][0, 7, 0].item() == 0
    assert current_out["selected_blocks"][0, 7, 0].item() == 3
    assert not torch.equal(oldest_out["hidden"][:, 7], current_out["hidden"][:, 7])


def test_sparse_selection_uses_block_importance_override() -> None:
    class PreferNewestBlock(ContentSparseTransformer):
        def _block_importance(
            self,
            query: torch.Tensor,
            block_summaries: torch.Tensor,
        ) -> torch.Tensor:
            block_count = block_summaries.shape[2]
            scores = torch.arange(block_count, device=query.device, dtype=query.dtype)
            return scores.reshape(1, 1, block_count).expand(query.shape[0], query.shape[1], -1)

    # Given: an override that always prefers the highest visible block id.
    model = PreferNewestBlock(
        vocab_size=17,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=8,
        n_hypotheses=3,
        block_size=2,
        top_k_blocks=1,
        window=2,
    ).eval()
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: the model emits sparse selection traces.
    with torch.no_grad():
        out = model(tokens)

    # Then: top-1 selection follows the override up to the causal boundary.
    assert torch.equal(out["selected_blocks"][0, :, 0], torch.arange(8) // 2)


def test_content_sparse_selection_recovers_decisive_prior_block() -> None:
    # Given: a hidden fixture where block 0 is the decisive match for token 4.
    model = ContentSparseTransformer(
        vocab_size=17,
        d_model=4,
        n_heads=1,
        n_layers=1,
        max_seq_len=6,
        n_hypotheses=3,
        block_size=2,
        top_k_blocks=1,
        window=2,
    )
    hidden = torch.zeros(1, 6, 4)
    hidden[0, 0:2, 0] = 10
    hidden[0, 4, 0] = 1

    # When: content-sparse selection scores the hand-shaped sequence.
    _, selected_blocks = model._select_blocks(hidden)

    # Then: token 4 recalls the decisive prior block instead of its current block.
    assert selected_blocks[0, 4, 0].item() == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"block_size": 0}, "block_size"),
        ({"top_k_blocks": 0}, "top_k_blocks"),
        ({"max_seq_len": 0}, "max_seq_len"),
        ({"d_model": 10, "n_heads": 4}, "d_model"),
        ({"attention_mode": "bad_mode"}, "attention_mode"),
    ],
)
def test_sparse_rejects_invalid_configuration(
    kwargs: dict[str, int | str],
    expected: str,
) -> None:
    # Given: a sparse model configuration with one invalid field.
    config = {
        "vocab_size": 17,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "max_seq_len": 8,
        "n_hypotheses": 3,
        "block_size": 2,
        "top_k_blocks": 3,
        "window": 2,
    } | kwargs

    # When / Then: construction fails at the model boundary with a named field.
    with pytest.raises(ValueError, match=expected):
        ContentSparseTransformer(**config)
