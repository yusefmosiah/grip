# Research docs index

This directory has two kinds of documents:

- `SPEC-*` files are preregistered experiment contracts. They define what must
  be true before a result can be interpreted.
- The mechanism docs below are the ingredient shelf. They collect external
  papers, open-source implementations, open-model mechanisms, and adapter
  policy before model code is written.

Read in this order:

1. `SPEC-000-derivative-probe.md` - the first gate: whether trajectory
   variables are already linearly readable from a dense backbone.
2. `SPEC-001-tasks-and-data.md` - the task/data contracts and leakage guards.
3. `SPEC-002-eval-protocol.md` - the milestone order and scoring discipline.
4. `SPEC-003-ablations-and-sweeps.md` - the causal A-vs-B comparison.
5. `MECHANISM_INGREDIENTS.md` - the current attention/mechanism inventory.
6. `OPEN_MODEL_MECHANISMS.md` - current open-model mechanisms worth borrowing.
7. `ADAPTER_STRATEGY.md` - how to import, copy, adapt, or defer external code.
8. `CODEX_DELEGATION_PLAN.md` - the autonomous work queue and commit boundaries.
9. `M_REGIME_PIVOT_G008.md` - the first headroom-gate decision and the next
   corrective trained-M-regime slice.
10. `M_REGIME_TRAINED_G010.md` - the minimal trained Bayesian M-regime result;
    still pivots, so Grip A/B remains unauthorized.
11. `M_REGIME_REVERSAL_G011.md` - the minimal trained reversal result; mixed
    across eight seeds, so Grip A/B remains unauthorized.
12. `M_REGIME_CALIBRATED_G013.md` - the trained Bayesian/reversal rerun with
    generated calibrated floors; still no program-level A/B authorization.
13. `M_REGIME_SWEEP_G017.md` - the reusable-runner rerun of the calibrated
    Bayesian/reversal sweeps; aggregate reports still withhold A/B
    authorization.
14. `M_REGIME_SCALE_G019.md` - a reversal seq-32/64 scale map through the
    module CLI; aggregate reports still pivot, so Grip A/B remains
    unauthorized.

The load-bearing experiment remains `grip-read (A)` vs `grip-select (B)` from
`SPEC-003`. The mechanism docs do not change the milestone order:
M-legibility, M-probe, M-regime, then M-avsb.

Current M-probe evidence supports the "add state" branch: derivative-supervised
backbones kept level probes readable while `d_conf`/`dd_conf` probes remained
near baseline across three seeds in the current local seq-128 robustness run.
Treat OLMo-Hybrid-style explicit state as the default design reference until a
stronger contrary probe result appears. Do not treat that seq-128 run as the
full seq-512/1024 preregistered sweep evidence.
