# Glossary

Authoritative terminology for the grip project. Code, configs, docs, and
variable names should match this. Where a prior draft used a different term
(mostly the old working name "heed"/"HEED"), it is mapped here for
translation and will be excised from committed material.

---

## The core term

**grip** — *noun* — a model's compact, queryable representation of its own
bearings on a task. *Not* a confidence score, *not* the hidden state, *not*
the logits, *not* the KV cache. It is a side state, trained to represent the
model's current relation to the problem: which hypotheses are live, whether
confidence is rising for good reasons, whether it is looping, whether the
frame is wrong, what should survive compaction.

> A model can be **confident with poor grip**, **uncertain with strong grip**,
> **fluent with poor grip**, or **hesitant with good grip**. Confidence is a
> scalar output property; grip is a multi-dimensional control state.

This is the central distinction of the program. The two are conflated
constantly in casual usage; do not conflate them here.

---

## Mechanism vs. state vs. summary

Three things get called "grip-something." Keep them straight:

- **grip state** (`G_t`) — the learned state vector at time *t* representing
  the model's current bearings. The object of interest.
- **grip summary** (`g_b`) — the per-block summary emitted by block *b*,
  trained with auxiliary losses to reconstruct known latents. The block-level
  unit that sparse attention selects over.
- **grip attention** / **grip-conditioned sparse attention** — the *mechanism*:
  sparse attention that selects content blocks *and* grip blocks, and uses the
  grip state to decide which content is worth reading.

---

## What lives inside a grip state

These are the component variables the grip state is trained to represent.
Grouped by function.

### Certainty-trajectory features (the load-bearing group)

The architectural translation of the Portfolio Mind claim that reasoning is
*meta-awareness of the temporal evolution* of one's belief, i.e. the
**derivatives of certainty**, not the certainty level.

- `d_conf` — **confidence slope**: first derivative of belief.
- `dd_conf` — **confidence acceleration**: second derivative of belief.
- *trajectory shape* — whether the confidence curve matches a recognized
  path toward resolution or confusion.

These are not arbitrary features. They are the direct translation of the
essay's thesis, and they are the thing a vanilla hidden state is hypothesized
*not* to retain. **Reconstruction of the derivative terms — not of the
level — is the actual test of the amnesia claim** (see "derivative probe"
below).

### Routing features (why the trajectory is what it is)

- `source_trust` — reliability state of the sources feeding the belief.
- `frame_stability` — whether the model believes it is solving the right
  problem.
- `conflict` — degree and type of contradiction among active sources.
- `grounding_gap` — how much confidence is driven by *evidence* vs. by
  *fluency*. This one has a free ground-truth label in synthetic tasks and
  is the natural guardrail against the self-amplification loop (see "failure
  modes" below).

### Control features (what to do next)

- `churn_risk` — probability that continued work is low-value looping.
- `progress_energy` — estimated payoff of more work.
- `tangent_pull` / `stash_priority` — estimated value of an off-path idea,
  and whether it should survive compaction.
- `ask_or_act` — continue / ask / verify / pivot / stop.

### Level features (the controls, not the experiment)

- `p_hyp` — distribution over live hypotheses.
- `entropy` — how diffuse the current belief is.
- `margin` — gap between top hypotheses.

These describe the *level* of belief. They will reconstruct trivially from a
hidden state because the model must compute them to predict well. Treating
their reconstruction as evidence for grip is the **level-probe fallacy** —
see below.

---

## The selection distinction (the whole point of the mechanism)

- **content-sparse attention** — the baseline. For a query, score content
  blocks by `q · k / √d` (relevance), select top-K, read them. This is what
  every existing sparse/KV-cache method does.
- **grip-conditioned sparse selection** — the key variant. The selection
  score adds an **epistemic-value** term:

  ```
  importance_b = f(query, content_repr_b) + λ · h(query_grip, grip_b)
  ```

  Grip drives *which content gets read*. This is distinguished from mere
  "grip-readable" (grip exists as extra context but `λ = 0`, so it does not
  influence selection). **The load-bearing experiment is the A-vs-B
  comparison: grip-readable (λ=0) vs. grip-conditioned (λ>0), at matched
  parameters and matched read budget.** If B does not beat A, the result is
  "auxiliary supervision reshapes representations" — not an attention result.

---

## Probes and the experiments that gate everything

- **level probe** — a linear probe asking whether the hidden state at step
  *t* can reconstruct `p_hyp` / `entropy` at step *t*. Will almost certainly
  *succeed*, and is therefore evidence for nothing. A positive result is
  expected and uninformative.
- **derivative probe** — a linear probe asking whether the hidden state can
  reconstruct `d_conf` / `dd_conf` (the *slope* and *acceleration* of belief).
  `d_conf` is **not** a function of `h[t]` alone — representing it requires
  the model to have *retained* its t−1 confidence in a readable form. This
  is the actual test of the amnesia claim. **Run this before building
  anything.** It bifurcates the architecture:
  - *derivative positive* → trajectory info is already in the state, just
    not routed to selection → minimal experiment is a frozen backbone + a
    selector head that wires it in.
  - *derivative negative* → the amnesia claim holds literally → grip state
    must be *added* to represent the trajectory; the full mechanism is
    motivated.

- **leakage-as-bottleneck** — the principle that grip must be able to
  reconstruct *meta*-variables (where the decisive evidence is, source trust)
  but **not** the content/label. Achieved by training the grip summary under
  an explicit capacity constraint. If grip can point to the decisive block
  but cannot decode what it says, any selection gain is *routing*, not
  *answer-passthrough*.

---

## Controls (the brutal ones)

- **generic memory baseline** — learned memory slots that are *not* grip
  (the `x-transformers` memory-token control). Rules out "any extra memory
  helps."
- **matched-parameter baseline** — same param count, no grip. Rules out
  "more capacity helps."
- **matched-compute baseline** — baseline is given the grip encoder's forward
  FLOPs as *extra content blocks to read*. The fair bar. Report a K/R-vs-accuracy
  curve, never a single operating point.
- **shuffled-grip ablation** — grip slots shuffled across samples / across
  time. If grip helps causally, shuffling must selectively hurt the tasks
  that need it.
- **wrong-sample grip ablation** — grip from a different sample. Same logic.

---

## Failure modes (designed against, not just warned about)

- **self-amplification loop** — the model attends to its own confidence and
  becomes more confident *because it attended*. Guardrail: train grip to
  predict `grounding_gap`; penalize confidence rising without posterior
  support; report calibration **as a function of grip-attention weight**.
  A positive slope on that curve is the failure signature. Pre-register it.
- **answer leakage** — supervised grip decodes the label. Guardrail: the
  bottleneck design above; lead with tasks where grip vars are orthogonal
  to the label (Source Reliability Reversal), and prove the orthogonality
  rather than asserting it.
- **generic-memory confound / compute confound / toy-world overfit** —
  handled by the controls above and by many generators + OOD evals.

---

## Things this project is *not* claiming

- Not consciousness, not emotion, not "a self." Only causally useful
  internal grip state, unless stronger evidence appears.
- Not that grip replaces content attention. Grip *conditions* content
  selection. Content is still what gets read.
- Not a benchmark-chasing result. The first success criterion is whether
  grip-conditioned sparse selection beats content-sparse at matched compute,
  with shuffled/wrong grip selectively harming the cases that need it.

---

## Deprecated terms (do not use; map if encountered)

| Old term | Use instead |
|---|---|
| heed / HEED / Heed | grip (mechanism) / the grip program |
| Epistemic Sparse Attention (ESA) | grip-conditioned sparse attention |
| Grip Attention (as a distinct brand) | grip attention / grip-conditioned sparse attention |
| "Day N" / six-day sprint / schedule | the milestone DAG / the sequence of events |
| posterior reconstruction as acceptance | derivative reconstruction as acceptance |

"Self-Model Attention," "Reflexive Sparse Attention," "Within-Attention,"
"Bearings," "Ken," "Waymark" were all candidate names considered and
discarded in the source program doc. They are not in use.
