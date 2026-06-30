# SPEC-001: Task suite & synthetic data design

> Framing (from the cognitive-transform pass): the generator is itself a reward
> model, and its failure mode is *legibility* — if d_conf is a local function
> of token identity, every downstream result is grammar-reading, not
> belief-tracking. The generator must preserve history-dependence.

## Design principles (non-negotiable)

1. **History-dependent belief dynamics.** The same token emitted at different
   points in different streams must be able to move belief by wildly different
   amounts. Bayesian update is multiplicative: `posterior ∝ likelihood · prior`,
   so a token's effect depends on the *entire* preceding posterior. The
   generator must not collapse this to a lookup table. (Transform 3.)
2. **Fixed sequence length, padded.** Never variable — pytorch #181213 MPS leak.
3. **Deterministic given seed.**
4. **Latents as first-class fields**, derived inside generate(), never post-hoc.
5. **Human-readable tiny vocab** (≤64). Readability > realism.

## Tasks (priority order)

### T0 — Bayesian Evidence Streams (the substrate; gate everything on this)

As specified in SPEC-000. K hypotheses, S sources with hidden reliability,
informative-vs-decoy emission, late reliability reversal. The decisive property
to verify: **d_conf is non-local** — a given (token, source) pair's belief-move
varies across positions and streams.

*Generator-instrumentation requirement:* emit, for debugging, the per-token
true belief-move alongside the token, so the bypass probe (SPEC-002 M-legibility)
has a target.

### T1 — Source Reliability Reversal (lead comparison task; orthogonal-to-label)

A focused variant of T0 where the load is *provenance*, not the answer:
- The decisive evidence for the answer is **early** and **from a source that
  later reverses** reliability.
- The label (argmax hypothesis) is determinable from the *aggregate* evidence
  and is NOT trivially decodable from source_trust alone.
- The interesting grip variables (`source_trust[t]`, decisive-index location)
  are orthogonal to the label by construction — verify this numerically:
  `mutual_info(source_trust, answer) ≈ 0`.

This is the A-vs-B comparison task because grip's claimed value (return to
claims from now-unreliable sources) is here, and leakage is hardest here.

### T2 — Churn / Agent-Slump Simulation (deferred)

Synthetic agent logs from a hidden Markov process (productive / looping /
wrong-frame / missing-info / breakthrough). Predict the meta-action
(continue / pivot / ask / stop / stash). Ground truth from the HMM state.
**Do not build until T0+T1 show signal.**

### T3 — Frame Switch, T4 Tangent Discovery, T5 Compaction Boundary (deferred)

Per the source program. Only after T0/T1/T2 establish the mechanism.

## Leakage hardening (applies to T0 and T1)

Three independent leakage channels; each needs a guard:

1. **Generator legibility** (Transform 3): d_conf must not be a local function
   of token identity. *Verified* by the bypass probe (SPEC-002 M-legibility).
2. **Supervision leakage**: aux supervision on posterior must not decode the
   label from a single grip vector. *Guarded* by the bottleneck (low-dim grip
   + reconstruction-of-meta-not-content) and *verified* by probing
   `grip → answer`: if R^2 > threshold, harden the task.
3. **Selection leakage**: the decisive block must be findable from grip but its
   *content* must not be. *Verified* by the bottleneck ablation.

## Acceptance for T0

- Posterior sanity-checked against a hand-computed 3-step example (closed form).
- Bypass probe fails to read d_conf off raw tokens above the noise floor
  (M-legibility gate). If it succeeds, harden the generator before proceeding.
- `mutual_info(source_trust, answer)` measured and logged; near-zero for T1.
- 10 samples human-printable; a reader can eyeball the belief trajectory.
