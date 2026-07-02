# G011 reversal trained M-regime

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G011 reversal result.

G011 ran the minimal trained source-reliability reversal task through the same
scorer-owned headroom gate introduced for G010. The first seed returned `keep`,
but the eight-seed local check was mixed:

- seed count: `8`
- `keep`: `3`
- `pivot`: `5`
- task: `reversal`
- sequence length: `16`
- vocabulary size: `64`
- hypotheses: `4`
- train batch size: `2`
- train steps: `2`
- learning rate: `0.001`
- device: `cpu`

Per-seed loss deltas (`content_sparse - dense`) were:

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | keep | 8.85194206237793 | 9.009800910949707 | 9.18521785736084 | 0.33327579498291016 |
| 1 | keep | 10.47003173828125 | 10.542250633239746 | 10.656399726867676 | 0.18636798858642578 |
| 2 | pivot | 8.141341209411621 | 8.07170581817627 | 8.035945892333984 | -0.10539531707763672 |
| 3 | pivot | 8.87148666381836 | 8.492606163024902 | 8.300219535827637 | -0.5712671279907227 |
| 4 | keep | 7.339022636413574 | 7.395559787750244 | 7.4411396980285645 | 0.10211706161499023 |
| 5 | pivot | 9.698992729187012 | 9.584622383117676 | 8.747310638427734 | -0.9516820907592773 |
| 6 | pivot | 9.14375114440918 | 9.345776557922363 | 8.811798095703125 | -0.3319530487060547 |
| 7 | pivot | 10.18399429321289 | 10.042388916015625 | 9.612290382385254 | -0.5717039108276367 |

The ignored artifacts live under `runs/m-regime-g011-reversal` for the first
seed and `runs/m-regime-g011-reversal-seeds` for the eight-seed check.

## Interpretation

The reversal task is more promising than the tiny Bayesian G010 run because
content-sparse underperforms dense on some seeds, creating possible headroom for
a later Grip selector. It is not yet a stable M-regime authorization. A/B would
be premature because the keep decision is not seed-stable, and the local
noise-floor artifact is still only a gate exercise, not a full preregistered
calibration.

Next work should strengthen the M-regime rather than add Grip state:

1. Add a real local noise-floor calibration runner for matched identical-config
   pairs instead of hand-authored local threshold JSON.
2. Evaluate reversal with a larger but still M1-friendly train budget.
3. Define the aggregate keep criterion across seeds before any Grip A/B branch
   can begin.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
