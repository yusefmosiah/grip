"""Batching utilities: StreamSample -> torch tensors."""
from __future__ import annotations
import numpy as np
import torch

from .streams import StreamSample


def collate(samples: list[StreamSample], device: str = "cpu") -> dict:
    """Stack a list of StreamSamples into batched torch tensors.

    Returns dict with keys:
        tokens:[B,T]  answer:[B]  posterior:[B,T,K]  entropy:[B,T]
        belief_move:[B,T]  d_conf:[B,T]  dd_conf:[B,T]
        source_idx:[B,T]  source_trust:[B,T,S]
        decisive_idx:[B,T]  real_mask:[B,T]  block_boundaries:[B,nb+1]
    """
    def stack(name, dtype):
        return torch.as_tensor(np.stack([getattr(s, name) for s in samples]), dtype=dtype)

    out = {
        "tokens": stack("tokens", torch.long),
        "answer": stack("answer", torch.long),
        "posterior": stack("posterior", torch.float32),
        "entropy": stack("entropy", torch.float32),
        "belief_move": stack("belief_move", torch.float32),
        "d_conf": stack("d_conf", torch.float32),
        "dd_conf": stack("dd_conf", torch.float32),
        "source_idx": stack("source_idx", torch.long),
        "source_trust": stack("source_trust", torch.float32),
        "decisive_idx": stack("decisive_idx", torch.long),
    }
    real_mask = np.stack([
        np.arange(samples[0].tokens.shape[0]) < int(s.metadata["natural_len"])
        for s in samples
    ])
    out["real_mask"] = torch.as_tensor(real_mask, dtype=torch.bool)
    out["block_boundaries"] = stack("block_boundaries", torch.long)
    return {k: v.to(device) for k, v in out.items()}


def make_batch(stream, n: int, seed: int = 0, device: str = "cpu") -> dict:
    """Convenience: generate n streams and collate."""
    samples = [stream.generate(seed=seed * 1000 + i) for i in range(n)]
    return collate(samples, device=device)
