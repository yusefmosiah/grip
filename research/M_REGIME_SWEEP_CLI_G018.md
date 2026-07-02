# G018 M-regime sweep CLI

Date: 2026-07-01

## Decision

G018 did not authorize Grip A/B. It made the reusable G016 runner callable as a
module CLI so later scale-map cells could be run through the same artifact
surface.

The implementation commit was `fd4cb2e` (`Add M-regime sweep CLI`). It extended
`src/grip/eval/m_regime_sweep.py` with CLI argument parsing and tests for
writing sweep artifacts from the command line.

## CLI Contract

The CLI exposes the task, sequence length, vocabulary/hypothesis parameters,
training budget, eval batch size, and output directory while preserving the
runner-owned calibration, seed loop, and aggregate summary logic.

## Interpretation

G018 is infrastructure, not headroom evidence. It enables G019 and later runs
to use the common runner path. Grip A/B remains unauthorized until a valid
aggregate cell satisfies the pre-declared M-regime rule.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
