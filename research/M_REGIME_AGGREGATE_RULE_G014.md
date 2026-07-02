# G014 aggregate M-regime decision rule

Date: 2026-07-01

## Decision

G014 did not authorize Grip A/B. It added the aggregate headroom decision rule
used to interpret seed-level M-regime results before any A-vs-B comparison can
start.

The implementation commit was `d88329a` (`Add aggregate headroom decision
rule`). It introduced `src/grip/eval/aggregate_headroom.py` and tests for the
aggregate gate.

## Rule

The aggregate M-regime gate is a pre-declared program-level rule over seed
decisions:

- at least eight seed-level decisions are required;
- every included seed must be interpretable under the scorer/noise-floor gate;
- `keep_rate >= 0.75` is required for program-level A/B authorization;
- otherwise the aggregate status is `pivot` or `blocked`.

This rule exists because isolated seed-level `keep` results are not sufficient
evidence that content-sparse underperforms dense in a stable headroom regime.

## Interpretation

G014 is a methodological gate, not an empirical result. It makes later records
such as G017, G019, G021, and G022 comparable by using one aggregate criterion.
No Grip A/B work is authorized by G014 alone.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
