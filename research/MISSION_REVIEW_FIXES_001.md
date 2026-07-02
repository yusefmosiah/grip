# MISSION: Review fixes 001 — make the gates trustworthy before any result is reported

Date: 2026-07-01
Origin: full-project deep review (models/training, eval/gates, tests/hygiene, methodology).
Status: OPEN. Each work item below is checked off with a commit reference when done.
Re-review happens after all P0 and P1 items are complete.

Governing principle: the honesty machinery (scorer authority, decision docs,
preregistration discipline) is the strongest part of the program. This mission
makes the experimental substrate worthy of it. No M-regime, probe, or A-vs-B
result may be cited as evidence until the P0 items are closed.

---

## P0 — Critical: gates currently produce untrustworthy decisions

### P0.1 Fix degenerate noise-floor calibration
**Status.** DONE in `ff8fa86` (calibration pairs now use distinct seeds,
thresholds derive from observed spread, all-zero spread fails loudly).
**Problem.** `noise_floor_calibration.py:143-152` runs left/right calibration
pairs with the *same config and same seed*. Runs are deterministic, so every
delta is exactly 0.0 and the threshold always falls back to the
`minimum_signal_floor=1e-6` epsilon (empirically verified). The keep gate is
effectively `sign(Δloss)` — the ≥8-seed machinery calibrates determinism, not
seed variance.
**Fix.**
- Calibration pairs must differ in seed (or data ordering) between left and
  right, drawn from the same seed *population* as decision runs but disjoint
  from the decision seed set.
- Threshold derives from the observed cross-seed spread (e.g. max or a
  quantile of |Δloss| across pairs), not an epsilon fallback. If all deltas
  are ~0, the calibration must FAIL loudly ("regime has no measurable
  variance — invalid"), not silently pass a 1e-6 floor.
- Update `tests/test_noise_floor_calibration.py` to assert nonzero spread on a
  real pair and assert the loud-failure path.

### P0.2 Bind noise-floor artifacts to decision-run configs
**Status.** DONE in `ff8fa86` (scorer blocks mismatched floors and reports the
mismatched config fields).
**Problem.** `load_noise_floor` never verifies the floor's embedded
calibration config matches the decision run (task, model dims, seq_len, batch
sizes, train_steps). A floor calibrated on any config gates any comparison.
"Interpretable" is a caller-supplied `preregistered=True` flag
(`score.py:92-97`) with no enforcement.
**Fix.** `load_noise_floor` (or `compare`) validates config identity between
the floor's calibration payload and each run's resolved config; mismatch →
blocked, with the differing fields named. Add a test with a mismatched floor.

### P0.3 Minimum-validity requirements for M-regime decisions
**Status.** DONE in `77f311c` (scorer recomputes valid-tier minimums,
smoke artifacts are unciteable, and aggregate summaries skip smoke rows).
**Problem.** `MRegimeConfig` defaults (`headroom_types.py:38-44`) are
`train_steps=0`, batch 1, seq_len 8; the sweep CLI inherits them. G010–G022
decisions compared near-random initializations on a single eval batch —
scientifically empty, yet emitted keep/pivot decisions.
**Fix.**
- The scorer refuses to emit keep/pivot (status → `blocked: below minimum
  training budget`) unless the run meets pre-declared minimums. Write the
  minimums in the SPEC-002 amendment (P1.1) first; suggested floor: models
  trained to a documented convergence criterion (or a fixed steps floor
  ≥ preregistered value), eval batch ≥ 8 sequences, seq_len ≥ the smallest
  preregistered cell (512).
- Smoke-tier runs remain allowed but their artifacts must carry
  `tier: smoke` and be unciteable by aggregate/authorization code
  (`aggregate_headroom`, `aggregate_summary` skip them with a logged reason).

### P0.4 Replace the `train/run.py` stub or fence it
**Status.** DONE in `9e21c55` (stub artifacts are only allowed under explicit
`mode: "stub-dry-run"`; `smoke` and `preregistered` are rejected before writes).
**Problem.** `train()` (`train/run.py:70-105`) writes `config.resolved.json`,
a hardcoded `dry_run` record with `loss: 0.0`, and a stub `eval_tensors.json`.
No model, no optimizer, no checkpoint. It accepts `mode: "preregistered"` and
`train.seed` is parsed but never applied — official-looking artifacts with
fabricated zero losses.
**Fix.** Either implement real training (reuse `headroom_training.py`'s loop;
apply `torch.manual_seed`/numpy seeding from `train.seed`; write real losses
and checkpoints), or make `train()` raise on any mode other than an explicit
`mode: "stub-dry-run"` until it does. No artifact with `loss: 0.0` and no
optimizer step may ever be labeled `preregistered`. Add a CPU test asserting
loss decreases over N steps (see P2.4).

### P0.5 Make the sparse model an actual selection mechanism
**Status.** DONE in `a4cadd7` (learnable selector scores now weight selected
context on the CE loss path; block summaries use compact full-block +
current-prefix tensors instead of the old per-token/per-block materialization;
`grip-read-A` and `grip-select-B` are runnable headroom variants with explicit
Grip state, where A reads Grip state without Grip selection and B adds Grip
state to selection scoring).
**Problem.** `ContentSparseTransformer` is not block-sparse attention:
- Selection happens once, after the full stack (`sparse.py:128-133`), via a
  parameter-free dot product (`_block_importance`, `sparse.py:112-119`);
  `topk` is non-differentiable and scores never enter the loss path —
  **selection is unlearnable**.
- The "read budget" adds one mean vector residually (`sparse.py:181-183`), so
  `top_k_blocks` is a nearly inert knob; read-budget sweeps over it are
  meaningless.
- `_summarize_blocks` (`sparse.py:209`) is O(B·T²·num_blocks·d) — a hidden
  quadratic inside the "sparse" model.
- The grip variants the hypothesis depends on (`grip-read-A`,
  `grip-select-B`) do not exist: `run.py:131-132` rejects them,
  `grip_state` is hardwired `None`, while `sweep_plan.py:174-177` already
  references them.
**Fix.**
- Rework selection so it gates token-level attention into selected blocks
  (per layer or per selection point), with learnable scoring on the loss path
  (straight-through / weighted-summary / Gumbel — pick one and document it).
- Replace the einsum summarization with a causal cumulative/block-prefix
  computation that is subquadratic.
- Implement `grip-read-A` and `grip-select-B` model variants and register
  them in `run.py`'s model factory. Grip state stops being `None`.
- Update `sparse.py`'s module docstring — until fixed it mischaracterizes the
  implementation ("the NSA/DSA selection surface").
- Keep/extend the prefix-causality and selector-counterfactual tests in
  `tests/test_sparse.py` against the new mechanism.

### P0.6 Add compute matching
**Status.** DONE in `8326337` (headroom eval artifacts now write compute
payloads; scorer-owned comparison reports include per-run compute,
`compute_tolerance`, and `compute_mismatches`; preregistered comparisons are
blocked on mismatched parameter count, estimated forward FLOPs, or read
budget).
**Problem.** README/GLOSSARY claim matched-compute comparisons; nothing in the
code measures or matches compute.
**Fix.** Add a FLOPs (or measured wall-clock + parameter-count) accounting
utility; every comparison report records per-model compute; `compare` flags
comparisons whose compute differs beyond a declared tolerance. The A-vs-B
contract requires matched compute by construction.

---

## P1 — Methodology: preregistration integrity

### P1.1 SPEC-002 amendment (write BEFORE the next decision run)
**Status.** DONE in `1a0ac56` (`SPEC-002-AMENDMENT-001.md`).
One dated amendment doc (`SPEC-002-AMENDMENT-001.md`) fixing, with numbers:
- The noise-floor procedure (per P0.1) and its failure mode.
- The aggregate keep criterion (currently 0.75 keep-rate, defined nine
  minutes after seeing the 3/8 data it adjudicates — re-declare it
  prospectively, with seed count ≥ 8 and the per-seed decision rule).
- Minimum training budget / eval batch / seq_len for a valid M-regime cell
  (feeds P0.3).
- The multi-task authorization rule: replace
  `authorize_avsb = any(task keeps)` (`aggregate_summary.py`) with a declared
  rule that does not grow more permissive as tasks are added (per-task
  authorization, or a corrected criterion).
- SPEC-000 probe thresholds: promote the working numbers (level-control
  R² ≥ 0.5, derivative-readable R² ≥ 0.2, bypass 0.05 / 1.6× chance) into
  the amendment or replace them with justified ones.

### P1.2 Terminology and spec-freezing discipline
**Status.** DONE in `8129232` (G-doc terminology now distinguishes
mid-program pre-declared rules from preregistration, G017 carries an erratum,
every M-regime decision doc records the current governing spec blob hashes, and
`SPEC-003-AMENDMENT-001.md` marks the post-probe add-state update).
- Stop calling mid-stream rules "preregistered"; use "pre-declared" and date
  them. Audit existing G-docs for the mislabel (G017 at minimum) and add
  errata notes — do not rewrite history.
- Freeze specs: record each spec's git blob hash in every decision doc that
  relies on it; amendments go in new dated files, never in-place edits.
  Retroactively note the post-probe add-state edit to SPEC-003 with an
  amendment marker.

### P1.3 Backfill missing decision docs
**Status.** DONE in `8129232` (`M_REGIME_AGGREGATE_RULE_G014.md`,
`M_REGIME_AGGREGATE_SUMMARY_G015.md`, `M_REGIME_RUNNER_G016.md`,
`M_REGIME_SWEEP_CLI_G018.md`, and `M_REGIME_HELDOUT_FIX_G020.md` were added and
indexed).
G014, G015, G016, G018, G020 exist only as commits/run dirs. G020 (the
`train_tokens[:1]` eval bug that superseded G017/G019) especially needs its
own record. Short docs are fine; absence is not.

### P1.4 Name the regime-search conditioning
**Status.** DONE in `8129232` (`SPEC-003-AMENDMENT-001.md` and
`research/README.md` now require future A-vs-B claims to be conditioned on the
searched-for headroom regime and to report the regime-search path).
Add a standing note (in SPEC-003 amendment or research/README): the headroom
regime, once found, will have been *selected for*; any A-vs-B claim is
conditioned on "in a regime constructed to have headroom," and the regime
search itself is reported alongside the result.

### P1.5 Rerun M-regime honestly
After P0.1–P0.3 land: run ONE honestly-trained M-regime cell at the smallest
preregistered size (1M params, seq 512, trained to the declared budget,
eval batch ≥ 8, ≥ 8 independent seeds) before any further gate refactoring.
Write its G-doc. This is the first M-regime evidence that counts.

---

## P2 — Statistical and correctness bugs

### P2.1 Seed hygiene
**Status.** DONE across `ff8fa86` and `df2d495`: `make_batch`, M-regime
training batches, noise-floor/sweep calibration seeds, probe/backbone seed
namespaces, and reversal labels are partitioned or RNG-derived without the
old seed modulo cycle.
- `make_batch` (`data/collate.py:44`): `seed*1000 + i` collides (seed=1,i=0 ==
  seed=0,i=1000) and train/eval disjointness rests on unvalidated invariants.
  Replace with a non-colliding scheme (e.g. hash/`SeedSequence`-spawned
  streams) or validate the invariants (`batch_size < 1000`,
  `eval_seed_offset > train_steps`) at construction with hard errors.
- `headroom_training.py:96-111`: sweep seed s and s+1 share all but one train
  batch (`range(seed, seed+steps)`) — "independent seeds" aren't. Give each
  sweep seed a disjoint batch-seed range.
- `run_probe_000.py` seed bases: train base `10M + seed*1M` collides with test
  base `20M + seed'*1M` across seeds; backbone-training generator seeds can
  reach probe bases for large `n_steps`. Partition the seed space explicitly
  and add an assertion test.
- `reversal.py:31`: `h_star = sample_seed % K` makes the label
  seed-derived and cyclic. Draw the label from the seeded RNG instead.

### P2.2 Padding
**Status.** DONE in `598acf0` (`real_mask` now flows through M-regime training
batches, dense/sparse forwards, masked next-token CE loss, local attention key
masks, and sparse block summaries; tests cover padded-loss equivalence,
dense masked-prefix equivalence, and padded block-summary exclusion).
`real_mask` exists in collate (`collate.py:33-37`) but nothing consumes it:
PAD=0 is attended normally, enters block summaries and selection, and padded
positions are included in train/eval cross-entropy
(`headroom_runs.py:57-63`, `headroom_training.py:139-146`), deflating losses
and diluting the dense-vs-sparse delta the gate tests. Mask padding in
attention, exclude PAD positions from block summaries/selection, and mask the
loss. Add a test: loss on a padded batch equals loss on the unpadded prefix.

### P2.3 Scorer/metrics fixes
**Status.** DONE across `f984584`, `2802949`, `fc90d35`, and `b8b6604`:
`compare()` now requires an explicit comparison output path; the `score.py` /
`train/run.py` `__main__` stubs are wired to real CLIs; no-decisive-position
recall is nullable; constant-target R² now distinguishes exact predictions
from noise floor; noise-floor staleness is keyed to a content hash;
first-token confidence moves are computed from the uniform prior; flip
provenance is jointly sorted; and the bypass probe now has a nondegenerate
positive control, window/ridge grid search, caller-owned RNG state, and answer
probe convergence reporting.
- `score.py:99`: `comparison.json` written to `runs[0].parent` — arbitrary
  location, silent overwrite. Take an explicit output path.
- `score.py:123`: `SystemExit("CODEX: wire CLI args")` — implement or delete
  the `__main__` stub. Same for `run.py:235-239`.
- `metrics.decisive_token_recall`: no-decisive-positions batch returns 0.0,
  indistinguishable from total failure — return `None`/NaN and have
  consumers skip it.
- `metrics.r2_score`: `ss_tot == 0` returns 0.0 even for exact predictions —
  distinguish constant-target from noise-floor.
- `bypass.py`: add a positive control (a synthetic *legible* target the same
  probe must decode) so a passing bypass gate certifies probe power, not
  probe weakness; sweep window size and ridge over a small grid; remove the
  hardcoded `torch.manual_seed(0)` in `_answer_accuracy` (thread caller seed)
  and add a convergence check to the 300-step answer probe.
- `noise_floor.py:80-83`: replace the exact-float staleness check with a
  content hash of the calibration payload.
- `streams.py:169-170` / `reversal.py:98-99`: `d_conf[0] = 0` hides genuinely
  decisive first tokens. Either compute the move from the uniform prior at
  t=0 or document the convention where decisive-token recall is defined.
- `streams.py:107-108`: sort `flip_steps` jointly with `flip_srcs` (sort
  pairs) so metadata provenance describes actual events.

### P2.4 Training-path test
**Status.** DONE in `0802204` (CPU test overfits a repeated tiny batch through
`train_model` and asserts cross-entropy decreases after real optimizer steps).
Add a CPU test asserting loss decreases over N optimizer steps on a small
model (guards P0.4's real trainer and `headroom_training.py`).

---

## P3 — Code quality and infrastructure

### P3.1 Deduplicate config plumbing
`MRegimeConfig` / `NoiseFloorCalibrationConfig` / `MRegimeSweepConfig` are
three hand-copied 15-field dataclasses with pure field-forwarding converters
(`_calibration_config`, `_m_regime_config`, `_headroom_config`) — a new field
silently defaults if one copy is missed (this already causes calibration pair
runs to emit misleading `m_regime_report.json` files with `status: blocked`).
Compose a shared base config; converters become trivial or disappear.
Similarly collapse the three provenance payload builders and drop the
redundant flat/nested duplicate keys in `_artifact_payload`.

### P3.2 Dedup models and data
`_Block` (`dense.py:18-48`) and `LocalCausalBlock`
(`sparse_components.py:41-75`) are near-identical — share one block class.
`streams.py`/`reversal.py` duplicate the Bayesian update and derivative
derivation (~40 lines) — extract a shared helper.

### P3.3 Kill dead code / wire promised code
**Status.** DONE across `41cf47d`, `2ab3bd4`, `eae3bba`, `4abcb92`,
`d576f1f`, and `4bc2b6e`: baseline naming now has one source of truth via
`headroom_baselines.py`, shared by noise-floor artifacts, calibration,
headroom run generation, and SPEC-003 sweep-plan runnable baseline variants;
stream-level `block_boundaries` were removed so configured model/eval
`block_size` is the only block authority; `sweep_plan.py` now marks the
SPEC-003 sweep matrix as declaration-only and rejects artifacts that claim
runner-consumable status; headroom runs now write report-only posterior
accuracy/NLL/Brier/ECE and source-answer MI metrics without making them
noise-floor gate requirements; and `configs/_README.md` now describes the
future YAML-template contract without claiming executable YAML templates or a
YAML loader already exist.

### P3.4 Model output typing
**Status.** DONE in `6ee9f5d` (`DenseTransformer` and
`ContentSparseTransformer` now return frozen output dataclasses, source
consumers use attribute access, and tests assert the typed output contract).
`forward` returning `dict[str, Tensor | SparseMetadata | None]` invites key
typos across eval code — return a dataclass/TypedDict.

### P3.5 Robustness niceties
**Status.** PARTIAL in `6ee9f5d` (dense now rejects sequences longer than
`max_seq_len`, matching sparse's existing boundary check).
- `dense.py:92`: validate `T <= max_seq_len` (sparse already does).
- Key decision logic on explicit identifiers, not `run_dir.name` string
  lookups (`_headroom_status`, `_losses`).

### P3.6 CI and test hygiene
- Add GitHub Actions running the CPU-only suite on Linux.
- `test_smoke.py:9-11`: hard MPS assert → `pytest.mark.skipif` (match
  `test_dense_model.py:43`); fill or delete the two stale
  "fill once streams.py is implemented" skips (`test_smoke.py:18,23`).
- Extend `test_repo_hygiene.py` to assert `runs/`, `*.pt`, checkpoints stay
  untracked.

---

## Sequencing

1. P1.1 amendment FIRST (rules dated before the runs they govern).
2. P0.1–P0.4 (gate trustworthiness), with P2.1/P2.2 folded in where they
   touch the same files.
3. P0.5–P0.6 (real mechanism + compute matching), then remaining P2.
4. P1.5 honest M-regime rerun + its G-doc.
5. P1.2–P1.4 doc debt and P3 cleanup in parallel throughout.
6. Request re-review.

## Done criteria

- All P0/P1 boxes closed with commit refs noted inline.
- Full suite green on CPU (locally and in CI).
- A noise-floor artifact with nonzero spread exists; a blocked-on-mismatch
  test exists; no artifact labeled `preregistered` was produced by a stub.
- One honestly-trained M-regime cell with its own decision doc.
