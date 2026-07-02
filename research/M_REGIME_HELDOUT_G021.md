# G021 heldout M-regime rerun

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G021 heldout result.

G020 fixed the M-regime runner so baseline losses are evaluated on a
deterministic heldout batch instead of `train_tokens[:1]`. G021 reran the prior
runner-backed cells with that corrected evaluation path:

- Bayesian seq 8
- reversal seq 16
- reversal seq 32
- reversal seq 64

The ignored artifacts live under `runs/m-regime-g021-heldout`. Each cell wrote a
calibrated noise-floor artifact, `summary.json`, per-seed decision artifacts,
and an aggregate report. The dense seed-0 artifact in each cell records:

- eval batch size: `1`
- eval seed: `10000`
- eval seed offset: `10000`

G017 and G019 are superseded for decision purposes because they were run before
the heldout eval fix. They remain useful historical records of the old runner
surface.

## Aggregate Results

| cell | seed count | keep count | keep rate | status | reason | authorize A/B |
| --- | ---: | ---: | ---: | --- | --- | --- |
| bayesian seq 8 | 8 | 3 | 0.375 | pivot | insufficient_keep_rate | false |
| reversal seq 16 | 8 | 2 | 0.25 | pivot | insufficient_keep_rate | false |
| reversal seq 32 | 8 | 0 | 0.0 | pivot | insufficient_keep_rate | false |
| reversal seq 64 | 8 | 1 | 0.125 | pivot | insufficient_keep_rate | false |

The aggregate gate still requires at least eight seeds, all interpretable, and a
keep rate of at least `0.75` before program-level A/B authorization.

## Bayesian Seq 8

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | keep | 3.9736385345458984 | 3.7172629833221436 | 4.410223960876465 | 0.4365854263305664 |
| 1 | pivot | 15.28010082244873 | 15.178909301757812 | 13.21197509765625 | -2.0681257247924805 |
| 2 | pivot | 12.38676929473877 | 12.214065551757812 | 11.343709945678711 | -1.0430593490600586 |
| 3 | pivot | 11.325509071350098 | 11.496027946472168 | 11.146881103515625 | -0.17862796783447266 |
| 4 | keep | 10.48399829864502 | 10.63675594329834 | 11.594584465026855 | 1.110586166381836 |
| 5 | pivot | 8.573334693908691 | 8.562079429626465 | 7.963428974151611 | -0.6099057197570801 |
| 6 | keep | 9.390128135681152 | 9.238896369934082 | 9.443093299865723 | 0.05296516418457031 |
| 7 | pivot | 9.048249244689941 | 9.09766960144043 | 7.907156467437744 | -1.1410927772521973 |

## Reversal Seq 16

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 9.035901069641113 | 8.975763320922852 | 8.936685562133789 | -0.09921550750732422 |
| 1 | pivot | 8.964920997619629 | 8.741857528686523 | 8.641946792602539 | -0.32297420501708984 |
| 2 | pivot | 10.586295127868652 | 10.49737548828125 | 10.47350788116455 | -0.11278724670410156 |
| 3 | pivot | 9.131284713745117 | 8.915578842163086 | 8.6677885055542 | -0.46349620819091797 |
| 4 | keep | 8.948930740356445 | 8.844473838806152 | 8.955339431762695 | 0.00640869140625 |
| 5 | pivot | 8.356484413146973 | 8.271915435791016 | 8.282998085021973 | -0.073486328125 |
| 6 | pivot | 10.13280963897705 | 10.119770050048828 | 9.836960792541504 | -0.2958488464355469 |
| 7 | keep | 8.797324180603027 | 8.824145317077637 | 8.825860977172852 | 0.02853679656982422 |

## Reversal Seq 32

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 13.187103271484375 | 13.125301361083984 | 12.928071975708008 | -0.2590312957763672 |
| 1 | pivot | 12.924471855163574 | 12.777276039123535 | 12.611823081970215 | -0.3126487731933594 |
| 2 | pivot | 9.2983980178833 | 9.382222175598145 | 9.24264144897461 | -0.055756568908691406 |
| 3 | pivot | 11.017172813415527 | 10.852025985717773 | 10.765267372131348 | -0.2519054412841797 |
| 4 | pivot | 13.954747200012207 | 13.96894359588623 | 13.361090660095215 | -0.5936565399169922 |
| 5 | pivot | 11.992681503295898 | 12.023874282836914 | 11.650882720947266 | -0.3417987823486328 |
| 6 | pivot | 9.62145709991455 | 9.475282669067383 | 9.442713737487793 | -0.1787433624267578 |
| 7 | pivot | 10.646527290344238 | 10.470014572143555 | 10.179621696472168 | -0.4669055938720703 |

## Reversal Seq 64

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 12.528679847717285 | 12.531037330627441 | 12.073125839233398 | -0.4555540084838867 |
| 1 | pivot | 11.526080131530762 | 11.38541316986084 | 10.786981582641602 | -0.7390985488891602 |
| 2 | keep | 10.257668495178223 | 10.315706253051758 | 10.442770957946777 | 0.1851024627685547 |
| 3 | pivot | 11.901375770568848 | 11.864877700805664 | 11.87442684173584 | -0.026948928833007812 |
| 4 | pivot | 12.078110694885254 | 12.073443412780762 | 11.724798202514648 | -0.35331249237060547 |
| 5 | pivot | 10.053466796875 | 10.129175186157227 | 10.052119255065918 | -0.0013475418090820312 |
| 6 | pivot | 8.593132972717285 | 8.525201797485352 | 8.320250511169434 | -0.27288246154785156 |
| 7 | pivot | 10.664966583251953 | 10.533491134643555 | 10.092862129211426 | -0.5721044540405273 |

## Interpretation

Heldout evaluation does not create stable content-sparse-underperforms-dense
headroom in the current tiny M-regime. Bayesian seq 8 has three seed-level
keeps but still fails the aggregate rule. Reversal becomes weaker under heldout
evaluation than the earlier in-sample records: seq 16 has two keeps, seq 32 has
none, and seq 64 has one.

Grip A/B remains unauthorized. The next implementation work should improve the
M-regime data/training surface or introduce a stronger dense-vs-sparse
diagnostic before any Grip state enters selection.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
