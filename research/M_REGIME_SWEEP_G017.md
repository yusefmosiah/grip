# G017 runner-backed calibrated M-regime

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G017 runner-backed result.

G017 reran the G013 calibrated trained M-regime checks through the reusable
`run_m_regime_sweep` path added in G016. The runner produced task-matched
noise-floor artifacts, per-seed `summary.json` rows, and aggregate reports using
the G014/G015 aggregate rule.

The ignored artifacts live under `runs/m-regime-g017-runner-backed`:

- Bayesian: `runs/m-regime-g017-runner-backed/bayesian`
- Reversal: `runs/m-regime-g017-runner-backed/reversal`

Both sweeps used:

- seed count: `8`
- train batch size: `2`
- train steps: `2`
- learning rate: `0.001`
- device: `cpu`

The aggregate rule requires at least eight seeds, all interpretable, and at
least a `0.75` keep rate before program-level A/B authorization.

## Bayesian Result

Bayesian remains a pivot across all eight seeds.

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 6.195436954498291 | 6.058210849761963 | 5.602477550506592 | -0.5929594039916992 |
| 1 | pivot | 7.271911144256592 | 7.321174621582031 | 5.248692035675049 | -2.023219108581543 |
| 2 | pivot | 5.6028642654418945 | 5.478436470031738 | 4.93002462387085 | -0.6728396415710449 |
| 3 | pivot | 8.142383575439453 | 8.025776863098145 | 6.809445858001709 | -1.3329377174377441 |
| 4 | pivot | 12.919876098632812 | 12.871122360229492 | 11.862272262573242 | -1.0576038360595703 |
| 5 | pivot | 8.795259475708008 | 8.351655960083008 | 7.121872901916504 | -1.673386573791504 |
| 6 | pivot | 8.978006362915039 | 8.957585334777832 | 8.830545425415039 | -0.1474609375 |
| 7 | pivot | 10.73857593536377 | 10.627118110656738 | 9.657743453979492 | -1.0808324813842773 |

Aggregate result:

- keep count: `0 / 8`
- keep rate: `0.0`
- status: `pivot`
- reason: `insufficient_keep_rate`
- authorize A/B: `false`

## Reversal Result

Reversal remains mixed across eight seeds.

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

Aggregate result:

- keep count: `3 / 8`
- keep rate: `0.375`
- status: `pivot`
- reason: `insufficient_keep_rate`
- authorize A/B: `false`

## Interpretation

The reusable runner reproduces the G013 scientific result while removing the ad
hoc script from the critical path. The calibrated Bayesian sweep has no
content-sparse-underperforms-dense headroom. The reversal sweep has minority
seed-level headroom, but it fails the pre-declared aggregate keep-rate gate.

Grip A/B remains unauthorized. The next implementation work should improve or
scale the M-regime baselines, not add Grip state or run `grip-read` versus
`grip-select`.

## Erratum: Terminology

Earlier wording in this record described the aggregate keep-rate gate as
"preregistered." The gate was introduced mid-program in G014 and should be read
as a dated pre-declared rule, not as a rule fixed before the G017 data existed.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
