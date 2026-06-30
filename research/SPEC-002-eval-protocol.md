# SPEC-002: Evaluation protocol

> Framing: this is a falsification apparatus, not a scoreboard. Its job is to
> reject the confound explanations (capacity / compute / leakage / generic
> memory) one at a time. "Grip wins" is meaningless unless the strongest version
> of every confound has been tuned as hard as grip and still loses (Transform 1).

## Milestone order (each gates the next)

```
M-legibility  →  M-probe  →  M-regime  →  M-avsb  →  M-noise-floor (already done) feeds all
```

### M-legibility — the gate in front of the gate

Before any model result is interpretable, show the **task** is not trivially
solvable from raw tokens.

- **Bypass probe:** fit `linear(raw_token_onehot[t-window:t] → d_conf[t])`.
  If R^2 exceeds the noise floor, the generator is too legible — *harden it,
  do not proceed to models.* No amount of model cleverness rescues a task that
  reads d_conf off grammar.
- Also probe `raw_tokens → answer`: if trivially high, T0 is broken.

Only when the bypass probe *fails* does M-probe mean anything.

### M-probe — the derivative probe (SPEC-000)

As specified. Frozen dense backbone; linear probes for level (control) and
derivative (experiment) targets. Bifurcates the architecture.

### M-regime — establish that selection errors exist

grip can only help where content-sparse measurably underperforms dense.
- Sweep (seq_len, decoy_density, num_blocks) and find the region where
  `content-sparse < dense` by a margin above the noise floor.
- **If no such region exists at our scale, STOP.** There is no headroom; any
  grip result is vacuous. Report this honestly — it's a finding, not a failure.

### M-noise-floor — measured first, pre-registered

- Same model, same config, **N≥8 seeds**, identical in every way.
- Record the *distribution of deltas between identical configs*. This is the
  ceiling below which any A-vs-B "win" is noise.
- Pre-register: seed count per comparison cell, and the minimum delta (in
  MSE / accuracy / Brier) that counts as signal. Write it down *before* looking
  at any A-vs-B numbers.

## Metric set (every cell reports all)

**Object:** accuracy, NLL, Brier, ECE.
**Grip (the mechanism witnesses):**
- posterior-recon error, source_trust-recon error (level/routing — controls),
- **d_conf-recon error** (derivative — the amnesia test),
- decisive-token recall.
**Selection quality (causal value of what was read):**
- value-per-attended-block (drop selected block, measure Δloss),
- bad-source revisit rate (on T1).

## Discipline rules (hard, not aspirational)

1. **The trainer never reports comparison numbers.** `eval/score.py` reads
   artifacts and decides who won. (Operating model: agents mis-report own work.)
2. **Controls get ≥ grip's tuning budget.** The matched-compute baseline's
   hyperparameter search is at least as wide as grip's. A stock baseline losing
   to a tuned grip is not a result.
3. **No single operating point.** Report the full K (content top-k) × R (grip
   top-r) curve. A win at one favorable (K,R) is cherry-picking.
4. **Noise floor above every delta.** No delta is reported as "signal" unless it
   exceeds the M-noise-floor ceiling for that metric.
5. **Specificity required for the central claim.** grip's gain must concentrate
   on the uncertainty/provenance tasks (T1), not be uniform. Uniform gain =
   "better head," not "grip-qua-epistemic." Report gain *by task type*.
6. **Seed count pre-registered.** No adding seeds after seeing results.
