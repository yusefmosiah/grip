"""Analysis: probes, attention inspection, causal ablation utilities.

The PROBE is gating experiment #0. Before building any grip mechanism, we ask:
is the certainty-trajectory info (d_conf, dd_conf) already present in the hidden
state of a vanilla model, or is it absent (the amnesia claim)?

  level probe (control): can h[t] reconstruct posterior/entropy at t?
      -> expected to SUCCEED; uninformative.
  derivative probe (the test): can h[t] reconstruct d_conf, dd_conf?
      -> if YES: info present, not routed to selection -> minimal grip = a wire.
      -> if NO:  amnesia holds literally -> grip state must be ADDED.

This bifurcates the entire downstream architecture. Run before building anything.
"""
from .bypass import BypassProbeConfig, BypassProbeResult, run_bypass_probe  # noqa: F401
from .probe import linear_probe, ProbeResult  # noqa: F401
