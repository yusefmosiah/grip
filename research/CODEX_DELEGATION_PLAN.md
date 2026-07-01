# Codex delegation plan

Status: active autonomous handoff as of 2026-07-01.

This plan turns the mechanism research into leaf workstreams for Codex agents.
It is downstream of the derivative-supervision robustness probe: level targets
stayed readable while `d_conf` and `dd_conf` stayed near baseline across three
seeds in the current local seq-128 robustness run, so the default branch is
**add explicit Grip state**. This is supportive branch evidence, not a complete
substitute for the seq-512/1024 preregistered sweep grid. The
wire-existing-derivatives branch is now a counterfactual to revisit only if a
stronger contrary probe result appears.

## Operating rules

- Commit and push after each slice. Do not batch unrelated work across slices.
- Get at least one independent review before each commit; use more reviewers
  for architecture, experiment-contract, or broad test-surface changes.
- Stage only the intended files. `uv.lock` is currently untracked and must stay
  out of unrelated commits.
- Keep `/Users/wiz/grip` as the harness and artifact authority.
- Stage 0 must run locally on CPU or MPS with PyTorch-only code paths. Do not
  require CUDA, Triton, FlashAttention, xFormers, or FlexAttention training for
  the first correctness result.
- Trainer code emits artifacts. Scoring code decides comparisons. No trainer
  path may self-report a winner.
- Treat `research/SPEC-*.md`, `research/README.md`, this plan, and tracked
  source/tests as authoritative. Treat ignored `runs/` files as local evidence
  only; regenerate any stale or invalid run artifact before using it.

## Evidence authority and known local caveats

- The active add-state evidence is the current derivative-supervised robustness
  run summary under `runs/probe-000-derivaux-w10/summary.json`. It supports the
  branch choice, but it was run at seq 128.
- `research/SPEC-000-derivative-probe.md` still describes the canonical dense
  probe shape at seq 512. Do not claim the seq-128 run completes that spec.
- `research/SPEC-002-eval-protocol.md` and `src/grip/eval/sweep_plan.py`
  require at least 8 seeds for preregistered comparisons.
- If `runs/spec-003/spec-003-sweep-plan.json` still says `seed_count: 5`, it is
  stale local output. Regenerate it from `default_spec003_plan()` before any
  autonomous sweep uses it.
- No full M-noise-floor artifact is currently part of the tracked contract. Any
  A-vs-B smoke can test plumbing, but it cannot claim signal until the
  noise-floor artifact exists and the delta exceeds it.

## Current decision surface

The next work is not to import a full open model. The open-model research is an
ingredient shelf: OLMo-Hybrid/Qwen3-Next inform the explicit-state branch, and
DeepSeek/NSA/DSA inform the top-K selection seam. The first runnable suite stays
local and inspectable.

The current dependency order is:

1. Make configs and artifact paths reproducible for the existing gates.
2. Fill enough trainer/scorer infrastructure that runs can emit and compare
   artifacts without interpretive shortcuts.
3. Pin the sparse/grip output contract with tests before implementing variants.
4. Implement local-only and content-sparse traces.
5. Generate or validate the M-noise-floor artifact with N>=8 seeds before any
   interpretable delta claim.
6. Run M-regime to prove content-sparse leaves headroom below dense.
7. Implement generic-memory, grip-read A, grip-select B, and causal controls.

Stop immediately and record a blocker if any of these occur:

- M-legibility shows raw-token leakage above the noise floor.
- Probe level controls fail, making derivative-probe interpretation invalid.
- The sweep plan fails validation or uses fewer than 8 seeds for an
  interpretable comparison.
- M-regime finds no content-sparse headroom below dense.
- Loss becomes NaN or artifacts are incomplete.
- A needed backend is CUDA-only on the local M-series path.

Smoke runs are allowed for plumbing. They must be labeled `smoke` and must not
be described as preregistered evidence.

## Slice 000: noise-floor authority gate

Owner: one implementation agent, one experiment-contract reviewer.

Primary files:

- `src/grip/eval/score.py`
- `src/grip/eval/sweep_plan.py`
- new tests under `tests/` if no existing test owns the noise-floor artifact

Tasks:

- Define the M-noise-floor artifact schema before broad comparisons:
  identical config pairs, seed list, metric deltas, per-metric ceiling, and
  minimum signal threshold.
- Require N>=8 seeds for any artifact that can authorize an interpretable
  A-vs-B or M-regime delta.
- Make `eval/score.py` mark comparisons `interpretable: false` when the
  referenced noise-floor artifact is absent, stale, or below the seed floor.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sweep_plan.py -q` passes.
- Add/run the focused scorer/noise-floor tests introduced in this slice.
- A temp comparison without a noise-floor artifact writes `interpretable:
  false`.
- A temp comparison with a valid N>=8 noise-floor fixture records the expected
  per-metric ceiling.

Commit boundary:

- Commit the noise-floor schema and scorer gate separately from model code.

## Slice 001: gate and artifact plumbing

Owner: one implementation agent, one reviewer.

Primary files:

- `src/grip/train/run.py`
- `src/grip/eval/score.py`
- `src/grip/eval/sweep_plan.py`
- `tests/test_sweep_plan.py`
- new config/artifact tests if needed

Tasks:

- Add a minimal config loader or config dataclasses for dense/probe/bypass/sweep
  runs. Keep it small; this is not a general experiment framework.
- Add the first YAML templates under `configs/`, because the directory currently
  contains only naming guidance.
- Make `train(config, run_dir)` create the run directory, persist the resolved
  config, write append-only logs, and expose enough metadata for later scoring.
- Make `score_run()` and `compare()` read run artifacts and write comparison
  JSON without claiming a Grip win unless preregistered metrics exist.

Acceptance:

- Focused tests pass for sweep/config/scoring surfaces.
- A tiny dry run writes deterministic artifact files under a temp directory.
- The trainer writes no winner claims.
- Reviewer confirms no experiment-contract weakening.

Minimum run artifact schema:

- `config.resolved.json`: resolved config, seed, device request, git SHA,
  model name, task name, size label, seq length, read budget, and whether the
  run is `smoke` or `preregistered`.
- `train.jsonl`: append-only step records with step, tokens, loss components,
  learning rate, elapsed time, and device.
- `eval_tensors.pt` or equivalent: logits/probs, targets, traces, and masks
  needed by `eval/score.py`.
- `metrics.json`: produced by `eval/score.py`, not by the trainer.
- `comparison.json`: produced only by `compare()`, with explicit
  `interpretable: false` if gates/noise floor/seed count are incomplete.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sweep_plan.py -q` passes.
- Run any new config/trainer/scorer tests added in this slice.
- Run a tiny temp dry run and assert `config.resolved.json`, `train.jsonl`, and
  scorer-owned `metrics.json` exist.
- Assert no trainer output contains `winner`, `wins`, or `beats`.

Commit boundary:

- Commit as a plumbing slice only. Do not include sparse model implementation.

Separate cleanup:

- The old probe deprecation warning cleanup is useful but must be its own small
  commit: update `run_probe_000.main()` so it no longer passes legacy
  `probe_epochs` or `device` arguments into the closed-form probe path, then
  run `PYTHONPATH=src uv run pytest tests/test_probe_runner.py tests/test_probe.py
  -q`.

## Slice 002: sparse trace contract

Owner: one implementation agent, one reviewer with model-output focus.

Primary files:

- `src/grip/models/sparse.py`
- `src/grip/models/__init__.py`
- new `tests/test_sparse.py`
- `src/grip/eval/metrics.py` only if trace semantics expose a real gap

Tasks:

- Define the shared sparse/grip output surface:
  `lm_logits`, `posterior`, `hidden`, `selected_blocks`, `selection_scores`,
  `grip_state`, and `grip_recon`.
- Add shape and mask tests for `selected_blocks` and `selection_scores`.
- Make causal masking explicit: no selector may select a future block.
- Keep `_block_importance()` as the swappable seam that Grip modifies later.
- Ensure decisive-token recall uses true position block ids, not selected index
  rank or local top-K order.
- Add lambda-isolation tests that can later prove variant A has no hidden path
  from Grip state into selection.

Acceptance:

- Tests prove output keys, tensor shapes, causal masks, and deterministic
  selection behavior on toy inputs.
- `ContentSparseTransformer` can be instantiated in a tiny config without
  requiring CUDA-only dependencies.
- Reviewer confirms A/B can share this contract without hidden selector paths.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sparse.py tests/test_dense_model.py -q`
  passes after `tests/test_sparse.py` is added.
- Instantiate the tiny sparse model in CPU mode and assert output keys exactly
  include the shared trace contract.
- Assert `selected_blocks.max()` never exceeds the current position's block id
  in a causal toy batch.

Commit boundary:

- Commit the contract and tests before adding Grip-read or Grip-select logic.

## Slice 003: local-only and content-sparse baselines

Owner: one implementation agent, one reviewer with performance/regression focus.

Primary files:

- `src/grip/models/sparse.py`
- `tests/test_sparse.py`
- `src/grip/train/run.py` only for model registry wiring
- `src/grip/eval/score.py` only for trace artifact loading

Tasks:

- Implement local-only attention using ordinary PyTorch masks or SDPA.
- Implement content-sparse top-K block selection from compressed block
  summaries.
- Emit `selected_blocks`, `selection_scores`, read budget, block size, and
  window metadata in artifacts.
- Keep NSA/DSA repositories as references only; do not vendor kernels in this
  slice.

Acceptance:

- Tiny CPU tests cover local-only, content-sparse, causal no-future selection,
  and selected-block recall on known decisive positions.
- Parameter/read-budget metadata is recorded for comparison.
- Reviewer confirms no CUDA/Triton/FlexAttention training dependency entered.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sparse.py tests/test_dense_model.py -q`
  passes.
- Run a tiny CPU forward pass for `local-only` and `content-sparse`; assert
  logits, selected blocks, selection scores, and read-budget metadata are
  present.
- Run `rg -n "triton|flash_attn|xformers|flex_attention" src/grip/models
  src/grip/train tests` and confirm no required training dependency was added.

Commit boundary:

- Commit local/content-sparse only. Do not include Grip state or A/B variants.

## Slice 004: M-regime headroom gate

Owner: one implementation agent, one analysis/review agent.

Primary files:

- `src/grip/train/run.py`
- `src/grip/eval/score.py`
- `src/grip/eval/sweep_plan.py`
- `runs/` artifacts are generated but should not be committed unless the repo
  already treats a small report as canonical.

Tasks:

- Require the Slice 000 M-noise-floor artifact before marking any M-regime
  comparison interpretable.
- Run the smallest local M-regime smoke that can compare dense, local-only, and
  content-sparse.
- Verify content-sparse underperforms dense by a margin above the relevant
  noise floor before spending work on Grip variants.
- Write a short run report if the result is interpretable; otherwise record the
  blocker and fix the harness/model.
- If only a smoke run is feasible, write `interpretable: false` and do not use
  the result to authorize A-vs-B claims.

Acceptance:

- Comparison JSON exists and is generated by `eval/score.py`.
- The report says keep/pivot/blocked using preregistered criteria, not intuition.
- Reviewer confirms M-avsb is not started unless M-regime shows headroom.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sweep_plan.py tests/test_dense_model.py
  tests/test_sparse.py -q` passes once sparse tests exist.
- Run the smallest temp M-regime comparison and assert `comparison.json` is
  generated by `eval/score.py`.
- If the noise-floor artifact is missing, assert `comparison.json` contains
  `interpretable: false`.

Commit boundary:

- Commit code changes and, if useful, a small tracked research/report summary.
  Do not commit ignored `runs/` artifacts or large checkpoints.

## Slice 005: explicit Grip state producer

Owner: one implementation agent, one architecture reviewer.

Primary files:

- `src/grip/models/sparse.py` or a new narrow module under `src/grip/models/`
- `tests/test_sparse.py`
- `src/grip/analysis/probe_training.py` only if aux-head reuse is cleaner than
  duplicating logic

Tasks:

- Add an explicit Grip state update path inspired by OLMo-Hybrid-style state
  layers plus attention refresh, scaled down to the toy model.
- Add reconstruction heads for posterior, entropy, `d_conf`, `dd_conf`, and
  source trust.
- Keep state production separate from selection scoring.
- Add leakage probes or hooks needed for `grip -> answer` noise-floor checks.
- Interpret OLMo-Hybrid-style narrowly here: explicit state update plus periodic
  attention refresh. Do not import OLMo code in Stage 0.

Acceptance:

- Tests prove the state producer exists, emits `grip_state` and `grip_recon`,
  and can be run with `lambda=0`.
- Reviewer confirms the implementation did not wire backbone hidden derivative
  probes as a shortcut.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sparse.py tests/test_probe_training.py
  -q` passes.
- Run a tiny CPU forward pass with `lambda=0`; assert `grip_state` and
  `grip_recon` are present while selection is unchanged from content-only.
- Assert reconstruction heads include `posterior`, `entropy`, `d_conf`,
  `dd_conf`, and `source_trust`.

Commit boundary:

- Commit state producer and tests only. Do not claim A/B performance.

## Slice 006: A-vs-B sweep implementation

Owner: one implementation agent, one experiment-contract reviewer, one final
gate reviewer.

Primary files:

- `src/grip/models/sparse.py`
- `src/grip/eval/sweep_plan.py`
- `src/grip/train/run.py`
- `src/grip/eval/score.py`
- `tests/test_sparse.py`
- `tests/test_sweep_plan.py`

Tasks:

- Implement `grip-read-A` with shared Grip state and `lambda=0`.
- Implement `grip-select-B` with `content_score + lambda * grip_score`.
- Implement shuffle-grip and wrong-sample-grip controls.
- Keep generic-memory matched on params, FLOPs, and read budget.
- Generate the preregistered `(K,R)` sweep plan before running broad cells.
- Split a smoke A/B run from the full preregistered grid. The overnight target
  may produce a smoke artifact; the full `{1M,4M,16M} x {512,1024} x K/R x 8`
  grid is a separate interpretable run.

Matched-compute rule:

- Always enforce equal read budget for content-sparse, A, and B.
- Record parameter counts for every variant.
- Until an explicit FLOP counter exists, require the generic-memory/control
  baselines to receive at least the same extra module budget as Grip and mark
  FLOP matching as provisional in comparison artifacts.
- Do not report a causal win until the comparison artifact records matched
  read budget, matched parameters or compensating capacity, and an explicit
  compute-accounting note.

Acceptance:

- A and B differ only in whether Grip enters selection scoring.
- Matched read budget and matched FLOPs checks remain enforced.
- Comparison uses T1 first, T0 as calibration sanity, and the full `(K,R)` curve
  once the smoke path is stable.
- Reviewers confirm no hidden content leak and no trainer-side winner claim.

QA:

- `PYTHONPATH=src uv run pytest tests/test_sparse.py tests/test_sweep_plan.py
  tests/test_probe_training.py -q` passes.
- Run a tiny smoke A/B comparison and assert `comparison.json` labels it
  `smoke` or `interpretable: false` unless all gates and noise-floor evidence
  are present.
- Assert A and B share identical Grip-state producer parameters and differ only
  in selector lambda/scoring configuration.

Commit boundary:

- Commit A/B implementation and tests separately from large sweep reports.

## Deferred lanes

- Full OLMo-Hybrid, Qwen3-Next, Mamba, or RWKV baselines.
- CUDA/Triton/FlashAttention/xFormers/FlexAttention training paths.
- Longformer, BigBird, Performer, Linformer, Nystromformer, Transformer-XL.
- H2O, SnapKV, StreamingLLM, PyramidKV KV-cache comparators.
- Differential attention, until Grip-select B exists and the project is ready
  to test distractor-cancellation explanations.

These lanes are valuable later, but they do not answer the first causal A-vs-B
question.

## Review packet for every slice

Each slice should leave:

- exact files changed;
- exact tests/commands run;
- artifact paths generated;
- reviewer names and verdicts;
- commit SHA and push result;
- explicit note that unrelated untracked files stayed uncommitted.

If a slice changes experiment semantics, get a second reviewer before commit.
If reviewers disagree, record the disagreement and choose the stricter
interpretation unless it conflicts with a preregistered spec.
