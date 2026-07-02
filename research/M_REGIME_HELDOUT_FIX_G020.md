# G020 heldout evaluation fix

Date: 2026-07-01

## Decision

G020 supersedes the decision value of G017 and G019. It fixed the M-regime
runner so evaluation losses are computed on a deterministic heldout batch
instead of the earlier `train_tokens[:1]` path.

The implementation commit was `a02e9cd` (`Add heldout M-regime eval split`).
It added explicit eval seed offset plumbing to the headroom runner,
calibration runner, sweep runner, and tests.

## Bug

The pre-G020 runner evaluated baseline losses on a slice of the training
tokens. That made G017 and G019 useful for runner/infrastructure debugging but
not valid as decision evidence for M-regime headroom.

## Fix

G020 introduced:

- heldout eval batch generation from `seed + eval_seed_offset`;
- persisted eval seed and eval seed offset metadata;
- tests that training and eval batches are disjoint by construction.

## Interpretation

G017 and G019 remain historical records, but they are superseded for decision
purposes. G021 is the first rerun through the corrected heldout eval surface.
Grip A/B remains unauthorized.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
