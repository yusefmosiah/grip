from __future__ import annotations

import torch

from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, make_batch
from grip.eval.headroom_training import (
    TrainingBatchRequest,
    TrainingLoopConfig,
    TrainingTokenBatch,
    next_token_loss,
    train_model,
    training_token_batches,
    training_tokens,
)
from grip.models import DenseTransformer


def test_training_tokens_uses_bayesian_evidence_stream() -> None:
    # Given: a small Bayesian M-regime token request.
    tokens = training_tokens(
        task="bayesian",
        seq_len=8,
        vocab_size=17,
        n_hypotheses=3,
        batch_size=2,
        seed=7,
        device="cpu",
    )

    # Then: the tokens match the public Bayesian stream generator exactly.
    stream = BayesianEvidenceStream(
        num_hypotheses=3,
        seq_len=8,
        vocab_size=17,
        seed=7,
    )
    expected = make_batch(stream, n=2, seed=7, device="cpu")["tokens"]
    assert torch.equal(tokens, expected)


def test_training_tokens_uses_reversal_stream() -> None:
    # Given: a small reversal M-regime token request.
    tokens = training_tokens(
        task="reversal",
        seq_len=16,
        vocab_size=64,
        n_hypotheses=4,
        batch_size=2,
        seed=11,
        device="cpu",
    )

    # Then: the tokens match the public reversal stream generator exactly.
    stream = SourceReliabilityReversalStream(seq_len=16, seed=11)
    expected = make_batch(stream, n=2, seed=11, device="cpu")["tokens"]
    assert torch.equal(tokens, expected)


def test_training_token_batches_use_deterministic_step_seeds() -> None:
    # Given: a small Bayesian multi-step training request.
    batches = training_token_batches(
        TrainingBatchRequest(
            task="bayesian",
            seq_len=8,
            vocab_size=17,
            n_hypotheses=3,
            batch_size=2,
            seed=5,
            steps=3,
            device="cpu",
        )
    )

    # Then: each optimizer step receives a deterministic seed range owned by the run seed.
    assert tuple(batch.seed for batch in batches) == (5_000_000, 5_000_001, 5_000_002)
    stream = BayesianEvidenceStream(num_hypotheses=3, seq_len=8, vocab_size=17, seed=5_000_001)
    expected_step_two = make_batch(stream, n=2, seed=5_000_001, device="cpu")["tokens"]
    assert torch.equal(batches[1].tokens, expected_step_two)


def test_training_token_batches_do_not_overlap_adjacent_run_seed_ranges() -> None:
    # Given: two adjacent decision seeds with the same training-step count.
    first = training_token_batches(
        TrainingBatchRequest(
            task="bayesian",
            seq_len=8,
            vocab_size=17,
            n_hypotheses=3,
            batch_size=1,
            seed=5,
            steps=3,
            device="cpu",
        )
    )
    second = training_token_batches(
        TrainingBatchRequest(
            task="bayesian",
            seq_len=8,
            vocab_size=17,
            n_hypotheses=3,
            batch_size=1,
            seed=6,
            steps=3,
            device="cpu",
        )
    )

    # Then: their generated batch seeds are disjoint.
    assert set(batch.seed for batch in first).isdisjoint(batch.seed for batch in second)


def test_train_model_decreases_loss_on_repeated_cpu_batch() -> None:
    # Given: a tiny dense model and an overfit-friendly repeated CPU batch.
    torch.manual_seed(0)
    tokens = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]] * 8, dtype=torch.long)
    model = DenseTransformer(
        vocab_size=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        n_hypotheses=2,
    )
    batches = tuple(
        TrainingTokenBatch(seed=seed, tokens=tokens.clone())
        for seed in range(20)
    )

    # When: the real optimizer-backed training loop runs.
    records = train_model(
        model=model,
        batches=batches,
        config=TrainingLoopConfig(dry_run_seed=0, lr=1e-2, vocab_size=8),
    )

    # Then: the training path performs optimizer steps that reduce CE loss.
    losses = [float(record["loss"]["total"]) for record in records]
    assert losses[-1] < losses[0] * 0.25


def test_next_token_loss_ignores_padded_targets() -> None:
    # Given: logits for one padded sequence and its unpadded prefix.
    torch.manual_seed(3)
    logits = torch.randn(1, 5, 7)
    tokens = torch.tensor([[1, 2, 3, 4, 0]])
    real_mask = torch.tensor([[True, True, True, True, False]])

    # When: next-token loss is computed with the real-token mask.
    padded_loss = next_token_loss(
        logits=logits,
        tokens=tokens,
        real_mask=real_mask,
        vocab_size=7,
    )
    prefix_loss = next_token_loss(
        logits=logits[:, :4],
        tokens=tokens[:, :4],
        real_mask=torch.ones((1, 4), dtype=torch.bool),
        vocab_size=7,
    )

    # Then: padding does not change the measured loss.
    assert torch.allclose(padded_loss, prefix_loss)
