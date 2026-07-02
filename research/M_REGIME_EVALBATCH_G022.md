# G022 heldout eval-batch stability check

Date: 2026-07-01

## Decision

Do not start Grip A/B from the G022 eval-batch result.

G021 reran the corrected heldout M-regime with `eval_batch_size=1`. G022 keeps
the same training budget and reruns the two closest cells with
`eval_batch_size=8`:

```bash
PYTHONPATH=src uv run python -m grip.eval.m_regime_sweep \
  runs/m-regime-g022-evalbatch8/bayesian-seq8 \
  --task bayesian --seq-len 8 --vocab-size 17 --n-hypotheses 3 \
  --train-steps 2 --train-batch-size 2 --eval-batch-size 8

PYTHONPATH=src uv run python -m grip.eval.m_regime_sweep \
  runs/m-regime-g022-evalbatch8/reversal-seq16 \
  --task reversal --seq-len 16 --vocab-size 64 --n-hypotheses 4 \
  --train-steps 2 --train-batch-size 2 --eval-batch-size 8
```

The ignored artifacts live under `runs/m-regime-g022-evalbatch8`. The dense
seed-0 artifact in both cells records:

- eval batch size: `8`
- eval seed: `10000`
- eval seed offset: `10000`

## Aggregate Results

| cell | seed count | keep count | keep rate | status | reason | authorize A/B |
| --- | ---: | ---: | ---: | --- | --- | --- |
| bayesian seq 8 | 8 | 1 | 0.125 | pivot | insufficient_keep_rate | false |
| reversal seq 16 | 8 | 1 | 0.125 | pivot | insufficient_keep_rate | false |

Both cells remain below the aggregate `0.75` keep-rate threshold.

## Bayesian Seq 8

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 8.050653457641602 | 7.877126216888428 | 7.608099937438965 | -0.4425535202026367 |
| 1 | pivot | 9.294023513793945 | 9.22972297668457 | 8.337301254272461 | -0.9567222595214844 |
| 2 | pivot | 8.391585350036621 | 8.385282516479492 | 7.941313743591309 | -0.4502716064453125 |
| 3 | pivot | 9.54677677154541 | 9.5414457321167 | 9.163554191589355 | -0.3832225799560547 |
| 4 | keep | 10.224770545959473 | 10.233702659606934 | 10.25505542755127 | 0.030284881591796875 |
| 5 | pivot | 8.658875465393066 | 8.517385482788086 | 8.053282737731934 | -0.6055927276611328 |
| 6 | pivot | 7.5921735763549805 | 7.576075553894043 | 7.3500542640686035 | -0.24211931228637695 |
| 7 | pivot | 8.011092185974121 | 7.8806471824646 | 7.535019874572754 | -0.4760723114013672 |

## Reversal Seq 16

| seed | status | dense loss | local loss | content-sparse loss | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | pivot | 10.34271240234375 | 10.295576095581055 | 10.043853759765625 | -0.298858642578125 |
| 1 | pivot | 11.04848861694336 | 11.069981575012207 | 10.776957511901855 | -0.2715311050415039 |
| 2 | pivot | 9.092510223388672 | 9.076684951782227 | 8.897583961486816 | -0.19492626190185547 |
| 3 | pivot | 9.976913452148438 | 9.756093978881836 | 9.4363374710083 | -0.5405759811401367 |
| 4 | keep | 9.699125289916992 | 9.72238826751709 | 9.70290756225586 | 0.0037822723388671875 |
| 5 | pivot | 9.608771324157715 | 9.667722702026367 | 9.304121017456055 | -0.30465030670166016 |
| 6 | pivot | 10.045412063598633 | 10.030272483825684 | 9.486831665039062 | -0.5585803985595703 |
| 7 | pivot | 10.20086669921875 | 10.113150596618652 | 9.915987014770508 | -0.2848796844482422 |

## Interpretation

The aggregate pivot is not an artifact of single heldout-sample noise. Increasing
the heldout eval batch from `1` to `8` reduces Bayesian seq 8 from three keeps
to one and reversal seq 16 from two keeps to one. The only remaining keeps have
small positive deltas, and neither cell approaches the aggregate keep-rate gate.

Grip A/B remains unauthorized. The next useful work should alter the task or
training surface, not proceed to Grip state.

## Spec Freeze

This decision was audited against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
