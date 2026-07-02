from __future__ import annotations

import numpy as np

from .streams import PAD_TOKEN, StreamSample


class SourceReliabilityReversalStream:
    """Focused SPEC-001 stream where an early decisive source later reverses."""

    K = 4
    S = 3
    vocab_size = 64

    def __init__(self, seq_len: int = 512, seed: int = 0) -> None:
        if seq_len < 16:
            msg = "seq_len must be at least 16 for a reversal schedule"
            raise ValueError(msg)
        self.T = seq_len
        self.seed = seed
        rng = np.random.default_rng(seed)
        info_vocab = self.vocab_size - 1
        eps = rng.normal(0.0, 0.55, size=(info_vocab, self.K))
        likelihood = np.exp(eps)
        self._likelihood = likelihood / likelihood.sum(axis=0, keepdims=True)
        self._marginal = self._likelihood.mean(axis=1)

    def generate(self, seed: int | None = None) -> StreamSample:
        sample_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(sample_seed)
        schedule_code = sample_seed // self.K
        label_slot = int(sample_seed % self.K)
        label_rng = np.random.default_rng(self.seed * 1_000_003 + schedule_code)
        h_star = int(label_rng.permutation(self.K)[label_slot])
        natural_low = max(12, (self.T * 3) // 4)
        natural_span = self.T - natural_low + 1
        natural_len = natural_low + int(schedule_code % natural_span)
        reversal_source = int(schedule_code % self.S)
        high = 0.88 + 0.04 * float((schedule_code // self.S) % 3)
        peer = 0.66 + 0.04 * float((schedule_code // (self.S * 3)) % 3)
        low = 0.08 + 0.04 * float((schedule_code // (self.S * 9)) % 3)
        reversal_step = max(
            6,
            (self.T // 3) + int(schedule_code % max(1, self.T // 12)),
        )
        early_start = max(1, self.T // 8)
        if early_start >= reversal_step:
            early_start = 1
        early_candidate_steps = list(range(early_start, min(reversal_step, early_start + 5)))
        pre_source_probs = np.full(self.S, (1.0 - 0.35) / (self.S - 1))
        pre_source_probs[reversal_source] = 0.35
        post_source_probs = np.full(self.S, (1.0 - 0.08) / (self.S - 1))
        post_source_probs[reversal_source] = 0.08

        tokens = np.full(self.T, PAD_TOKEN, dtype=np.int64)
        source_idx = np.full(self.T, -1, dtype=np.int64)
        posterior = np.empty((self.T, self.K), dtype=np.float64)
        source_trust = np.empty((self.T, self.S), dtype=np.float64)
        prior = np.full(self.K, 1.0 / self.K, dtype=np.float64)

        for t in range(natural_len):
            trust = np.full(self.S, peer, dtype=np.float64)
            trust[reversal_source] = high
            if t >= reversal_step:
                trust[reversal_source] = low
            source_trust[t] = trust
            if t in early_candidate_steps:
                src = reversal_source
            elif t < reversal_step:
                src = int(rng.choice(self.S, p=pre_source_probs))
            else:
                src = int(rng.choice(self.S, p=post_source_probs))
            source_idx[t] = src
            informative = t >= early_start and (
                t in early_candidate_steps or rng.random() < trust[src]
            )
            if informative:
                tok_dist = self._likelihood[:, h_star]
                tok = int(rng.choice(self.vocab_size - 1, p=tok_dist)) + 1
            else:
                tok_dist = self._marginal
                tok = int(rng.choice(self.vocab_size - 1, p=tok_dist)) + 1
            tokens[t] = tok
            likelihood = (
                trust[src] * self._likelihood[tok - 1]
                + (1.0 - trust[src]) * self._marginal[tok - 1]
            )
            post = prior * likelihood
            post = post / post.sum()
            posterior[t] = post
            prior = post

        if natural_len < self.T:
            tokens[natural_len:] = PAD_TOKEN
            posterior[natural_len:] = posterior[natural_len - 1]
            source_trust[natural_len:] = source_trust[natural_len - 1]

        topmass = posterior.max(axis=1)
        log_posterior = np.zeros_like(posterior)
        np.log(posterior, out=log_posterior, where=posterior > 0)
        entropy = -(posterior * log_posterior).sum(axis=1)
        d_conf = np.zeros(self.T, dtype=np.float64)
        d_conf[0] = topmass[0] - (1.0 / self.K)
        d_conf[1:] = topmass[1:] - topmass[:-1]
        dd_conf = np.zeros(self.T, dtype=np.float64)
        dd_conf[1] = topmass[1] - (2.0 * topmass[0]) + (1.0 / self.K)
        dd_conf[2:] = topmass[2:] - 2 * topmass[1:-1] + topmass[:-2]
        decisive_idx = (np.abs(d_conf) >= 0.02).astype(np.int64)
        early_decisive_steps = [
            t for t in early_candidate_steps
            if decisive_idx[t] == 1 and source_idx[t] == reversal_source
        ]
        block_size = max(1, self.T // 16)
        block_boundaries = np.concatenate([
            np.arange(0, self.T, block_size),
            [self.T],
        ]).astype(np.int64)

        return StreamSample(
            tokens=tokens,
            answer=h_star,
            posterior=posterior,
            entropy=entropy,
            belief_move=d_conf.copy(),
            d_conf=d_conf,
            dd_conf=dd_conf,
            source_idx=source_idx,
            source_trust=source_trust,
            decisive_idx=decisive_idx,
            block_boundaries=block_boundaries,
            metadata={
                "h_star": h_star,
                "natural_len": natural_len,
                "reversal_source": reversal_source,
                "reversal_step": reversal_step,
                "early_decisive_steps": early_decisive_steps,
                "block_size": block_size,
            },
        )
