from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BeliefLatents:
    entropy: np.ndarray
    d_conf: np.ndarray
    dd_conf: np.ndarray
    decisive_idx: np.ndarray


def source_aware_update(
    *,
    prior: np.ndarray,
    token: int,
    source_trust: float,
    likelihood_table: np.ndarray,
    marginal: np.ndarray,
) -> np.ndarray:
    likelihood = (
        source_trust * likelihood_table[token - 1]
        + (1.0 - source_trust) * marginal[token - 1]
    )
    posterior = prior * likelihood
    return posterior / posterior.sum()


def posterior_latents(
    posterior: np.ndarray,
    *,
    prior_topmass: float,
    decisive_threshold: float = 0.02,
) -> BeliefLatents:
    topmass = posterior.max(axis=1)
    log_posterior = np.zeros_like(posterior)
    np.log(posterior, out=log_posterior, where=posterior > 0)
    entropy = -(posterior * log_posterior).sum(axis=1)
    d_conf = np.zeros(topmass.shape[0], dtype=np.float64)
    if topmass.shape[0]:
        d_conf[0] = topmass[0] - prior_topmass
        d_conf[1:] = topmass[1:] - topmass[:-1]
    dd_conf = np.zeros(topmass.shape[0], dtype=np.float64)
    if topmass.shape[0] > 1:
        dd_conf[1] = topmass[1] - (2.0 * topmass[0]) + prior_topmass
        dd_conf[2:] = topmass[2:] - 2 * topmass[1:-1] + topmass[:-2]
    decisive_idx = (np.abs(d_conf) >= decisive_threshold).astype(np.int64)
    return BeliefLatents(
        entropy=entropy,
        d_conf=d_conf,
        dd_conf=dd_conf,
        decisive_idx=decisive_idx,
    )
