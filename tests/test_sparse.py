import pytest
import torch
import torch.nn.functional as F

from grip.models import ContentSparseTransformer
from grip.models.sparse_components import CausalBlockSummaries


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


def test_grip_variants_return_explicit_state() -> None:
    # Given: grip-read and grip-select variants.
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
    read_model = _tiny_sparse_model(attention_mode="grip_read").eval()
    select_model = _tiny_sparse_model(attention_mode="grip_select").eval()

    # When: each variant runs a forward pass.
    with torch.no_grad():
        read_out = read_model(tokens)
        select_out = select_model(tokens)

    # Then: both expose explicit grip state and reconstruction tensors.
    assert read_out["metadata"]["attention_mode"] == "grip_read"
    assert select_out["metadata"]["attention_mode"] == "grip_select"
    assert read_out["grip_state"].shape == (1, 8, 16)
    assert read_out["grip_recon"].shape == (1, 8, 2)
    assert select_out["grip_state"].shape == (1, 8, 16)
    assert select_out["grip_recon"].shape == (1, 8, 2)


def test_grip_read_uses_content_selection_like_content_sparse() -> None:
    # Given: content-sparse and grip-read models with identical parameters.
    torch.manual_seed(29)
    content_model = _tiny_sparse_model(attention_mode="content_sparse").eval()
    read_model = _tiny_sparse_model(attention_mode="grip_read").eval()
    read_model.load_state_dict(content_model.state_dict())
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: both modes select sparse blocks.
    with torch.no_grad():
        content_out = content_model(tokens)
        read_out = read_model(tokens)

    # Then: Grip A changes the read state, not the selector.
    assert torch.equal(content_out["selection_scores"], read_out["selection_scores"])
    assert torch.equal(content_out["selected_blocks"], read_out["selected_blocks"])


def test_sparse_selection_is_causal_at_block_transitions() -> None:
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
    # Given: local and content-sparse models with identical weights.
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


def test_content_sparse_selector_scores_receive_loss_gradient() -> None:
    # Given: a content-sparse model with learnable selector projections.
    torch.manual_seed(19)
    model = _tiny_sparse_model()
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: next-token loss flows through selected sparse context.
    out = model(tokens)
    loss = F.cross_entropy(
        out["lm_logits"][:, :-1].reshape(-1, model.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()

    # Then: selector parameters are on the loss path.
    assert model.selector_query.weight.grad is not None
    assert model.selector_key.weight.grad is not None
    assert model.selector_query.weight.grad.abs().sum().item() > 0
    assert model.selector_key.weight.grad.abs().sum().item() > 0


def test_grip_read_state_receives_next_token_loss_gradient() -> None:
    # Given: a grip-read model whose selector remains content-only.
    torch.manual_seed(23)
    model = _tiny_sparse_model(attention_mode="grip_read")
    tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)

    # When: next-token loss flows through the Grip read context.
    out = model(tokens)
    loss = F.cross_entropy(
        out["lm_logits"][:, :-1].reshape(-1, model.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()

    # Then: the explicit Grip state and reconstruction head are trained by the read path.
    assert model.grip_state_projection.weight.grad is not None
    assert model.grip_recon.weight.grad is not None
    assert model.grip_recon_projection.weight.grad is not None
    assert model.grip_state_projection.weight.grad.abs().sum().item() > 0
    assert model.grip_recon.weight.grad.abs().sum().item() > 0
    assert model.grip_recon_projection.weight.grad.abs().sum().item() > 0


def test_block_summaries_match_compact_causal_prefix_means() -> None:
    # Given: hidden states with two-token blocks.
    model = _tiny_sparse_model()
    hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)

    # When: block summaries are computed.
    summaries = model._summarize_blocks(hidden)

    # Then: full block means and current-block prefixes are represented compactly.
    expected_full = torch.tensor(
        [[[1.0, 2.0], [5.0, 6.0], [8.0, 9.0]]]
    )
    expected_current_prefix = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0], [4.0, 5.0], [5.0, 6.0], [8.0, 9.0]]]
    )
    assert torch.equal(summaries.full, expected_full)
    assert torch.equal(summaries.current_prefix, expected_current_prefix)
    assert torch.equal(summaries.token_blocks, torch.tensor([0, 0, 1, 1, 2]))


def test_block_summaries_ignore_padding_mask() -> None:
    # Given: hidden states whose last block contains one real token and one PAD token.
    model = _tiny_sparse_model()
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [100.0, 100.0]]])
    real_mask = torch.tensor([[True, True, True, False]])

    # When: block summaries are computed with the real-token mask.
    summaries = model._summarize_blocks(hidden, real_mask=real_mask)

    # Then: the padded vector is excluded from full-block and prefix summaries.
    expected_full = torch.tensor([[[2.0, 2.0], [5.0, 5.0]]])
    expected_current_prefix = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [5.0, 5.0]]])
    assert torch.equal(summaries.full, expected_full)
    assert torch.equal(summaries.current_prefix, expected_current_prefix)


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
            block_summaries: CausalBlockSummaries,
        ) -> torch.Tensor:
            scores = torch.zeros(
                query.shape[0],
                query.shape[1],
                block_summaries.full.shape[1],
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
            block_summaries: CausalBlockSummaries,
        ) -> torch.Tensor:
            block_count = block_summaries.full.shape[1]
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
