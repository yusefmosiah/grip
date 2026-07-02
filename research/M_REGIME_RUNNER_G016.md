# G016 reusable M-regime sweep runner

Date: 2026-07-01

## Decision

G016 did not authorize Grip A/B. It moved repeated M-regime cells onto the
reusable sweep runner used by later records.

The implementation commit was `d180df1` (`Add M-regime sweep runner`). It added
`src/grip/eval/m_regime_sweep.py` and tests for producing calibrated
noise-floor artifacts, per-seed decisions, and aggregate summaries from one
runner path.

## Runner Contract

The runner is responsible for:

- generating the task-matched noise-floor calibration for a cell;
- running the configured seed set;
- writing per-seed scorer artifacts;
- writing `summary.json` with the G014/G015 aggregate decision.

## Interpretation

G016 reduces script drift. It does not change the evidence state: A/B remains
blocked until a runner-backed cell satisfies the pre-declared aggregate rule
under valid training/eval budgets.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
