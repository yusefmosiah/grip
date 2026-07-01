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

The load-bearing experiment remains `grip-read (A)` vs `grip-select (B)` from
`SPEC-003`. The mechanism docs do not change the milestone order:
M-legibility, M-probe, M-regime, then M-avsb.

Current M-probe evidence supports the "add state" branch: derivative-supervised
backbones kept level probes readable while `d_conf`/`dd_conf` probes remained
near baseline across three seeds. Treat OLMo-Hybrid-style explicit state as the
default design reference until a stronger contrary probe result appears.
