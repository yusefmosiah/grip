# G019 reversal M-regime scale map

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G019 reversal scale result.

G019 used the G018 module CLI to rerun the promising reversal branch at longer
toy sequence lengths:

```bash
PYTHONPATH=src uv run python -m grip.eval.m_regime_sweep \
  runs/m-regime-g019-reversal-scale/seq32 \
  --task reversal --seq-len 32 --vocab-size 64 --n-hypotheses 4 \
  --train-steps 2 --train-batch-size 2

PYTHONPATH=src uv run python -m grip.eval.m_regime_sweep \
  runs/m-regime-g019-reversal-scale/seq64 \
  --task reversal --seq-len 64 --vocab-size 64 --n-hypotheses 4 \
  --train-steps 2 --train-batch-size 2
```

Both runs wrote calibrated noise-floor artifacts, sequence-level `summary.json`
files with per-seed rows, per-seed decision reports, and aggregate reports under
`runs/m-regime-g019-reversal-scale`.

These are still tiny CPU M-regime checks, not the full SPEC grid. They are
useful for choosing the next implementation slice; they do not authorize
`grip-read` versus `grip-select`.

## Seq 32 Result

Seq 32 has no seed-level keeps.

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 10.681900978088379 | 10.472051620483398 | 9.888920783996582 | -0.7929801940917969 |
| 1 | pivot | 13.188545227050781 | 13.102836608886719 | 12.313243865966797 | -0.8753013610839844 |
| 2 | pivot | 10.192601203918457 | 10.03265380859375 | 9.741594314575195 | -0.4510068893432617 |
| 3 | pivot | 11.30152416229248 | 10.996441841125488 | 10.562134742736816 | -0.7393894195556641 |
| 4 | pivot | 9.753388404846191 | 9.64085865020752 | 9.597417831420898 | -0.15597057342529297 |
| 5 | pivot | 12.81053638458252 | 12.798663139343262 | 12.574531555175781 | -0.23600482940673828 |
| 6 | pivot | 11.557967185974121 | 11.550339698791504 | 11.227911949157715 | -0.33005523681640625 |
| 7 | pivot | 10.726664543151855 | 10.63955020904541 | 10.189736366271973 | -0.5369281768798828 |

Aggregate result:

- seed count: `8`
- interpretable count: `8`
- keep count: `0 / 8`
- keep rate: `0.0`
- status: `pivot`
- reason: `insufficient_keep_rate`
- authorize A/B: `false`

## Seq 64 Result

Seq 64 has two seed-level keeps, still far below the aggregate gate.

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 8.648305892944336 | 8.534438133239746 | 8.092183113098145 | -0.5561227798461914 |
| 1 | pivot | 13.404097557067871 | 13.520281791687012 | 13.113022804260254 | -0.2910747528076172 |
| 2 | keep | 10.355260848999023 | 10.332963943481445 | 10.466402053833008 | 0.11114120483398438 |
| 3 | keep | 10.227335929870605 | 10.219675064086914 | 10.25791072845459 | 0.030574798583984375 |
| 4 | pivot | 12.873431205749512 | 12.866782188415527 | 12.46595287322998 | -0.40747833251953125 |
| 5 | pivot | 10.885760307312012 | 10.845566749572754 | 10.51052474975586 | -0.37523555755615234 |
| 6 | pivot | 9.43790340423584 | 9.400638580322266 | 9.220763206481934 | -0.21714019775390625 |
| 7 | pivot | 11.48140811920166 | 11.486800193786621 | 11.081130027770996 | -0.40027809143066406 |

Aggregate result:

- seed count: `8`
- interpretable count: `8`
- keep count: `2 / 8`
- keep rate: `0.25`
- status: `pivot`
- reason: `insufficient_keep_rate`
- authorize A/B: `false`

## Interpretation

Longer toy reversal contexts did not make content-sparse-underperforms-dense
headroom seed-stable. Seq 32 removed the minority keeps seen at seq 16. Seq 64
kept only two seeds, below the aggregate rule's `0.75` keep-rate threshold.

This points away from Grip A/B implementation and toward improving the M-regime
itself. The next useful work should change the task or training surface in a
way that can make dense reliably stronger than content-sparse before any Grip
state enters the selector.
