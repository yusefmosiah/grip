# G010 minimal trained M-regime

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G010 minimal trained result.

G010 replaces the G008 untrained synthetic smoke tokens with a small trained
Bayesian M-regime pass. Dense, local, and content-sparse baselines now share:

- task: `bayesian`
- seed: `0`
- train batch size: `2`
- train steps: `2`
- learning rate: `0.001`
- device: `cpu`
- scorer-owned loss metric and local loss noise floor

The local ignored artifact run at `runs/m-regime-g010-trained` returned:

- `status`: `pivot`
- `authorize_avsb`: `false`
- `interpretable`: `true`
- `comparison_reason`: `ok`
- dense loss: `6.195436954498291`
- local loss: `6.058210849761963`
- content-sparse loss: `5.602477550506592`

Content-sparse did not underperform dense by more than the loss noise floor. It
outperformed dense in this tiny run, so Grip A/B remains causally uninformative.

## Interpretation

This is the intended corrective result for G009: before adding Grip state, the
baseline sparse comparison must first show usable headroom. G010 makes that
comparison more honest by using real task-generated Bayesian tokens, matched
training budget metadata, and per-baseline seeded initialization. It still does
not establish the M-regime headroom required by SPEC-002.

The result is not a kill decision. The run is intentionally tiny and exists to
gate the next autonomous slice. The next work should improve the baseline
M-regime before any A-vs-B branch work:

1. Add a reversal-task trained pass under the same scorer-owned gate.
2. Increase the Bayesian task budget only after the artifact contract remains
   stable.
3. Keep Grip A/B unauthorized until content-sparse loses to dense beyond the
   preregistered noise floor.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
