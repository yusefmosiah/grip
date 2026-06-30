"""Bayesian Evidence Streams — the first task.

Hidden state: one of K hypotheses is true. The stream emits evidence tokens
from sources with varying reliability, plus decoys, contradictions, and late
reliability reversals. The model must infer the posterior.

Ground truth is computed by forward-simulating Bayesian belief update under the
generative model, so every latent is known exactly at every step.

DESIGN CONSTRAINTS (non-negotiable):
  - Deterministic given a seed. Same seed => identical stream.
  - Human-readable tiny vocab (<=256 tokens).
  - Fixed sequence length per sample (pad if needed). REQUIRED: the MPS
    allocator leaks under varying shapes (pytorch issue #181213). The generator
    must never emit variable-length sequences.
  - Emit d_conf / dd_conf as first-class fields, not derived post-hoc.

CODEX TASK: implement generate() per the spec in
research/SPEC-000-derivative-probe.md. The mathematical update rule and the
exact token grammar are specified there; do not invent them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StreamSample:
    """One generated evidence stream with full ground-truth latents.

    Shapes: T = sequence length (fixed), K = num hypotheses.
    """
    tokens: np.ndarray            # int[T]      token ids in [0, vocab_size)
    answer: int                   #             argmax hypothesis (the label)
    # --- level features (trivial probe targets; the control) ---
    posterior: np.ndarray         # float[T, K]  P(h_k | evidence_{<=t})
    entropy: np.ndarray           # float[T]     Shannon entropy of posterior
    # --- derivative features (THE probe targets; the actual test) ---
    d_conf: np.ndarray            # float[T]     posterior[t]-posterior[t-1] (top mass)
    dd_conf: np.ndarray           # float[T]     second difference of top mass
    # --- routing features ---
    source_trust: np.ndarray      # float[T, num_sources]  reliability per source
    # --- ground-truth indices for the decisive-token-recall metric ---
    decisive_idx: np.ndarray      # int[T]       1 at steps that moved belief most
    # --- block boundaries for sparse attention ---
    block_boundaries: np.ndarray  # int[num_blocks+1]  inclusive-start indices
    metadata: dict = field(default_factory=dict)


class BayesianEvidenceStream:
    """Generates Bayesian Evidence Streams.

    Args:
        num_hypotheses: K.
        num_sources: number of evidence sources (each with hidden reliability).
        seq_len: fixed T. Padded to this length.
        vocab_size: <=256, tiny and human-readable.
        seed: deterministic.
    """

    def __init__(
        self,
        num_hypotheses: int = 4,
        num_sources: int = 3,
        seq_len: int = 512,
        vocab_size: int = 64,
        seed: int = 0,
    ):
        self.K = num_hypotheses
        self.num_sources = num_sources
        self.T = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def generate(self, seed: int | None = None) -> StreamSample:
        """Produce one stream. Raises NotImplementedError until CODEX implements."""
        raise NotImplementedError(
            "CODEX: implement per research/SPEC-000-derivative-probe.md. "
            "Forward-simulate Bayesian belief update; emit all latents."
        )
