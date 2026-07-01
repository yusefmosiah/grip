from __future__ import annotations

import numpy as np

from .streams import StreamSample


def format_stream_sample(sample: StreamSample, max_steps: int = 10) -> str:
    """Render a compact human-readable trace for SPEC-001 sample inspection."""
    natural_len = int(sample.metadata["natural_len"])
    shown_steps = min(max_steps, natural_len, sample.tokens.shape[0])
    header = f"answer={sample.answer} natural_len={natural_len}"
    lines = [header]
    for t in range(shown_steps):
        top_h = int(sample.posterior[t].argmax())
        top_p = float(sample.posterior[t, top_h])
        trust = np.array2string(sample.source_trust[t], precision=2, separator=",")
        lines.append(
            f"t={t:03d} tok={int(sample.tokens[t]):02d} src={int(sample.source_idx[t])} "
            f"top={top_h}:{top_p:.3f} move={float(sample.belief_move[t]):+.4f} "
            f"trust={trust}",
        )
    return "\n".join(lines)
