from __future__ import annotations

import numpy as np
import torch

from grip.data import (
    SourceReliabilityReversalStream,
    format_stream_sample,
)
from grip.data.collate import collate
from grip.eval.metrics import mutual_info_discrete


def test_reversal_stream_marks_early_decisive_source_that_later_reverses():
    # Given: a focused source-reliability reversal stream.
    stream = SourceReliabilityReversalStream(seq_len=96, seed=3)

    # When: a sample is generated.
    sample = stream.generate(seed=11)

    # Then: early decisive steps are from the source whose reliability reverses.
    reversal_source = sample.metadata["reversal_source"]
    reversal_step = sample.metadata["reversal_step"]
    decisive_steps = sample.metadata["early_decisive_steps"]
    assert decisive_steps
    assert max(decisive_steps) < reversal_step
    assert all(sample.source_idx[t] == reversal_source for t in decisive_steps)
    assert all(sample.decisive_idx[t] == 1 for t in decisive_steps)
    assert sample.source_trust[0, reversal_source] > sample.source_trust[-1, reversal_source]
    assert sample.answer == int(sample.posterior[-1].argmax())


def test_reversal_stream_posterior_update_uses_source_reliability():
    # Given: a focused source-reliability reversal stream and sample.
    stream = SourceReliabilityReversalStream(seq_len=96, seed=3)
    sample = stream.generate(seed=11)

    # When: the posterior update is recomputed from token and source trust.
    t = sample.metadata["early_decisive_steps"][0]
    tok = int(sample.tokens[t])
    src = int(sample.source_idx[t])
    trust = float(sample.source_trust[t, src])
    prior = sample.posterior[t - 1]
    source_likelihood = trust * stream._likelihood[tok - 1] + (1.0 - trust) * stream._marginal[tok - 1]
    expected = prior * source_likelihood
    expected = expected / expected.sum()

    # Then: the emitted posterior matches the source-aware Bayesian update.
    np.testing.assert_allclose(sample.posterior[t], expected, atol=1e-12)


def test_reversal_stream_source_trust_answer_mi_is_near_zero():
    # Given: many samples from the focused T1 task.
    stream = SourceReliabilityReversalStream(seq_len=96, seed=17)
    samples = [stream.generate(seed=seed) for seed in range(240)]
    trust_features = []
    for sample in samples:
        reversal_step = sample.metadata["reversal_step"]
        natural_len = sample.metadata["natural_len"]
        pre = sample.source_trust[:reversal_step].mean(axis=0)
        post = sample.source_trust[reversal_step:natural_len].mean(axis=0)
        trust_features.append(np.rint(np.concatenate([pre, post]) * 100).astype(np.int64))
    _, trust_signature = np.unique(np.stack(trust_features), axis=0, return_inverse=True)
    answers = np.asarray([sample.answer for sample in samples], dtype=np.int64)

    # When: MI(source_trust, answer) is measured.
    mi = mutual_info_discrete(torch.as_tensor(trust_signature), torch.as_tensor(answers))

    # Then: source-trust trajectory is orthogonal to the label.
    assert mi < 0.02


def test_collate_includes_source_idx_and_belief_move():
    # Given: generated reversal samples.
    stream = SourceReliabilityReversalStream(seq_len=48, seed=23)
    samples = [stream.generate(seed=seed) for seed in range(4)]

    # When: they are collated.
    batch = collate(samples)

    # Then: the debugging/provenance latents remain available in batch form.
    assert batch["source_idx"].shape == (4, 48)
    assert batch["belief_move"].shape == (4, 48)
    np.testing.assert_allclose(
        batch["belief_move"].numpy(),
        np.stack([sample.d_conf for sample in samples]),
        atol=1e-12,
    )


def test_format_stream_sample_prints_ten_readable_steps():
    # Given: a generated sample.
    stream = SourceReliabilityReversalStream(seq_len=64, seed=29)
    sample = stream.generate(seed=5)

    # When: a ten-step report is formatted.
    report = format_stream_sample(sample, max_steps=10)

    # Then: a reader can inspect token, source, belief, move, and trust.
    lines = report.splitlines()
    step_lines = [line for line in lines if line.startswith("t=")]
    assert len(step_lines) == 10
    assert "answer=" in lines[0]
    assert all("tok=" in line for line in step_lines)
    assert all("src=" in line for line in step_lines)
    assert all("move=" in line for line in step_lines)
    assert all("trust=[" in line for line in step_lines)
