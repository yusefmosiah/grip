"""Data generators for grip.

Every generator MUST emit, per timestep, ground-truth latent epistemic state:
  - posterior over hypotheses        (the *level* — trivial probe target)
  - entropy                          (the *level* — trivial probe target)
  - confidence slope  d_conf[t]      (the *derivative* — the real probe target)
  - confidence accel  dd_conf[t]     (the *derivative* — the real probe target)
  - source_trust state               (routing variable)
  - decisive_evidence_idx            (for decisive-token recall metric)

The level vs. derivative distinction is load-bearing — see GLOSSARY.md.
The generator is where the synthetic-stream advantage is realized: these
labels are free here and unobtainable in natural language.
"""
from .inspect import format_stream_sample  # noqa: F401
from .reversal import SourceReliabilityReversalStream  # noqa: F401
from .streams import BayesianEvidenceStream, StreamSample  # noqa: F401
from .collate import collate, make_batch  # noqa: F401
