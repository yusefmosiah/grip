# G008 M-regime pivot

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G008 smoke result.

The Slice 004 headroom gate ran dense, local, and content-sparse smoke baselines
with a valid local loss noise-floor artifact. The scorer marked the comparison
interpretable, but the headroom report returned:

- `status`: `pivot`
- `authorize_avsb`: `false`
- dense loss: `14.784398078918457`
- local loss: `10.219420433044434`
- content-sparse loss: `10.554800987243652`

This does not satisfy the SPEC-002 M-regime requirement that content-sparse
underperform dense by more than the relevant noise floor. Starting Grip A/B from
this result would test architecture plumbing, not the causal claim.

## Interpretation

The G008 run is a smoke gate, not a trained M-regime. It proves the comparison
plumbing, scorer authority, noise-floor gating, and keep/pivot/blocked reporting
work. It does not prove the real regime is absent.

The pivot is most likely caused by the synthetic untrained setup: one fixed
token sequence, randomly initialized models, and cross-entropy on next-token
logits. That surface can make dense look worse than local/content-sparse for
reasons unrelated to sparse selection failures.

## Next Slice

Run a minimal trained M-regime before any Grip A/B work:

1. Train dense, local, and content-sparse on the same small Bayesian/reversal
   task, seed, token budget, and device.
2. Evaluate with scorer-owned metrics and a valid loss/accuracy noise-floor
   artifact.
3. Require content-sparse to underperform dense by more than the noise floor
   before enabling Grip A/B.
4. If trained content-sparse still does not underperform dense, pivot away from
   Grip A/B and improve the task or sparse baseline first.

This keeps the program aligned with SPEC-002: M-regime must establish headroom
before M-avsb can be meaningful.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
