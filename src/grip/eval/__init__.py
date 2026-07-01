"""Metrics, ablations, and the score script.

DISCIPLINE (from the operating model): the trainer never reports its own
comparison numbers. All metrics are computed by score.py, which reads run
artifacts and writes the comparison. 'Who won' is computed separately from
who ran the job.

Required metrics (per GLOSSARY and the program):
  Object:    accuracy, NLL, Brier, ECE
  Grip:      posterior-reconstruction error, source-trust recon error,
             d_conf recon error (the derivative probe!), decisive-token recall
  Control:   value-per-attended-block, selected-block causal value
"""
from .metrics import (  # noqa: F401
    accuracy, nll, brier_score, ece,
    recon_error, decisive_token_recall, mutual_info_discrete,
)
from .sweep_plan import default_spec003_plan, validate_sweep_plan  # noqa: F401
