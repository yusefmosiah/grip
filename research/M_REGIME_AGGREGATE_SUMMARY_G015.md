# G015 aggregate M-regime summary reports

Date: 2026-07-01

## Decision

G015 did not authorize Grip A/B. It added durable aggregate summary artifacts so
multi-seed M-regime results can be audited without manually reading every
per-seed run directory.

The implementation commit was `7854356` (`Add aggregate summary reports`). It
introduced `src/grip/eval/aggregate_summary.py` and tests for summary payloads.

## Artifact Contract

Aggregate summaries record:

- task/cell metadata;
- seed count and keep count;
- keep rate;
- aggregate status and reason;
- whether A/B is authorized;
- the seed-level rows used to compute the aggregate.

Smoke-tier or otherwise unciteable rows are skipped rather than silently
participating in an authorization decision.

## Interpretation

G015 is an auditability step. It makes the G014 aggregate rule visible in
machine-readable artifacts, but it is not itself evidence for headroom. Grip
A/B remains unauthorized until an honestly trained, interpretable aggregate
cell satisfies the pre-declared rule.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
