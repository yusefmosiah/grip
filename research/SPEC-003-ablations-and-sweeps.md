# SPEC-003: Ablation matrix & sweep grid

> Framing: ablations are *backdoor-closing interventions* on a causal graph,
> not a checklist of "what if we removed this feature." Each ablation must close
> a specific non-causal path from tokens→answer. If it doesn't close a path,
> it's interpretability, not causal identification — defer it. (Transform 4.)

## The causal claim and its backdoors

**Claim:** the `grip_state → selection → attended → answer` path is causal
and carries epistemic (not merely capacity) value.

**Backdoor paths to close:**

| Path | What it would mean | Closing intervention |
|---|---|---|
| `tokens → hidden → grip → answer` (grip leaks content) | grip is a content side-channel | **Bottleneck:** grip reconstructs *where* decisive evidence is, not *what* it says. Verify `grip→answer` probe is at noise floor. |
| `aux → grip` AND `aux → hidden → answer` (supervision reshapes repr) | gain is from aux supervision, not selection | **λ=0 (variant A):** identical aux losses, grip readable but not selection-driving. |
| `read-budget → answer` (more reads = better) | grip wins by reading more | **Matched budget + matched FLOPs:** baseline gets grip-encoder FLOPs as extra content blocks. |
| `grip ← correct sample` (grip only works with the right sample's state) | grip isn't causal, just correlated | **Shuffle / wrong-sample grip:** grip from a different (n) or shuffled in time. |

## The experiment matrix (core; everything else is deferred)

Rows = variants. All at **matched params and matched compute**; all tuned with
**≥ grip's budget**; all at **≥ the pre-registered seed count**; reported as a
**(K,R) curve**, not a point.

| Variant | selection | grip read | aux sup | purpose |
|---|---|---|---|---|
| **dense** | full | — | no | upper reference |
| **local-only** | window | — | no | cheap floor |
| **content-sparse** | top-K content | — | no | the stock baseline grip replaces |
| **generic-memory** | top-K content | generic mem slots | no | capacity confound |
| **grip-read (A)** | top-K content | grip, λ=0 | yes | supervision-reshapes-repr confound |
| **grip-select (B)** | top-K content + λ·grip | grip | yes | **the mechanism** |
| **grip-select + shuffle-grip** | as B but grip shuffled | grip | yes | causal-use-of-correct-state |
| **grip-select + wrong-sample-grip** | as B but grip from n'≠n | grip | yes | same |
| **grip-select + bottleneck-off** | as B, grip unconstrained | grip | yes | leakage check (B should still beat w/o bottleneck only if non-leaky) |

**The load-bearing comparison is A vs B**, at matched params, matched compute,
matched read budget, ≥ pre-registered seeds, on T1 (Source Reliability Reversal),
with the full (K,R) curve, above the noise floor.

After the robustness probe, both A and B should share the same explicit
added-state Grip module. A/B differ only in whether that state enters selection:
A reads/reconstructs Grip with `lambda=0`; B uses `lambda>0` in the selector.

If B does not beat A → the result is "aux supervision reshapes representations"
(not an attention result). Report and stop the attention line; consider the
calibration/pivot path.

## Sweep grid (the smallest grid that can't be dismissed)

- **Sizes:** 1M, 4M, 16M. (64M only if 16M is stable and signal present.)
- **Seeds:** ≥ the M-noise-floor-derived count per cell (likely 5–8).
- **Seq lengths:** 512, 1024. (2048 only if memory + the regime map allow.)
- **Read budget curve:** K ∈ {4, 8, 16}, R ∈ {2, 4} (for grip variants).
- **Tasks:** T1 first (lead comparison). T0 as calibration sanity. T2 only on signal.

**Explicitly NOT in the first sweep:** every-model-family reproduction, the 12
toy architectures, compaction tasks, tangent tasks. These are the program's
later stages; running them now burns budget without answering the gating question.

## Reporting (per cell, per sweep)

A one-page memo (the "report after every sweep") containing:
1. The (K,R) curves for A vs B, with seed-variance bands, **with the noise-floor
   ceiling drawn as a horizontal line.**
2. Gain decomposed by task type (specificity check).
3. The leakage probes: `grip→answer` R^2, bypass-probe R^2.
4. Decisive-token recall; value-per-attended-block.
5. Calibration (Brier, ECE) — and calibration **as a function of grip-attention
   weight** (the self-amplification-loop signature; a positive slope = failure).
6. A keep/pivot/kill call against the pre-registered criteria.
