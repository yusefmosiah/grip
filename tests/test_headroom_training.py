from __future__ import annotations

import torch

from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream, make_batch
from grip.eval.headroom_training import TrainingBatchRequest, training_token_batches, training_tokens


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

    # Then: each optimizer step receives the deterministic next training seed.
    assert tuple(batch.seed for batch in batches) == (5, 6, 7)
    stream = BayesianEvidenceStream(num_hypotheses=3, seq_len=8, vocab_size=17, seed=6)
    expected_step_two = make_batch(stream, n=2, seed=6, device="cpu")["tokens"]
    assert torch.equal(batches[1].tokens, expected_step_two)
