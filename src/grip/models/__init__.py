"""Attention mechanisms for grip.

Order of implementation (mirrors the milestone DAG, not a calendar):
  1. dense       — full causal transformer (quality upper reference)
  2. local       — sliding-window only (cheap baseline)
  3. sparse      — local + top-K content-block selection (the stock baseline)
  4. grip_read   — sparse + grip-readable but lambda=0 (control A)
  5. grip_select — sparse + grip-conditioned selection lambda>0 (variant B)

The A-vs-B comparison at matched params and matched read budget is the
load-bearing experiment. See GLOSSARY.md and research/SPEC-001-avsb.md.
"""
from .dense import DenseTransformer          # noqa: F401
from .sparse import ContentSparseTransformer  # noqa: F401
