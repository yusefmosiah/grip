# grip

> **grip** — *noun* — a model's compact, queryable representation of its own
> bearings on a task: which hypotheses are live, whether its confidence is
> rising for good reasons, whether it is looping, whether a frame is wrong,
> whether a tangent is fertile, and what should survive compaction.

**grip** is a research program asking a single, sharp question:

> Does sparse attention get better — at matched compute — when it is guided by
> a learned representation of the model's own *grip on the task*, rather than
> only by content similarity and recency?

## The claim, in one paragraph

Standard sparse attention selects which context blocks to read by *content
relevance*. It never selects by *epistemic value* — by how much reading a block
would improve the model's bearings. **Grip attention** maintains a compact
learned state of the model's own task grip (uncertainty, confidence
**trajectory**, source reliability, frame stability, churn risk, tangent
value), lets the model attend to that state the way it attends to content, and
uses that state to decide which content is worth reading. The hypothesis is
that this is the missing layer between *reasoning* (trajectory search, which
models already do well) and *understanding* (a stable, queryable model of the
model's own relation to the task, which they don't have).

The mechanism gap is real and well-defined: every existing sparse-attention and
KV-cache selection method (H2O, SnapKV, NSA/DSA, the survey literature) scores
blocks on *content* importance. None wires a supervised epistemic state into
the **selection** mechanism itself. That conjunction is grip's white space.

## The essay this descends from

grip is the architectural translation of [The Portfolio Mind](https://mosiah.org/articles/the-portfolio-mind/),
whose load-bearing technical claim is that current LLMs are **architecturally
amnesiac about their own certainty trajectories**:

> Each token generation experiences the full weather system of competing
> expectations, but only the final barometric reading — the compressed hidden
> states — carries forward.

The essay argues that what we call "System 2 reasoning" is **meta-awareness of
the temporal evolution of one's own interference patterns** — i.e., tracking
the *derivatives* of certainty (slope, acceleration, recognized trajectory),
not the certainty level itself. grip is the attempt to give a model exactly
that, in a form it can attend over and select with.

## Status

**Research phase — gated implementation.** The data, dense baseline, probe,
bypass, metric, and sweep-contract surfaces are in place. The first probe gate
has supportive evidence for the add-state branch: derivative-supervised dense
backbones kept level probes readable while `d_conf`/`dd_conf` remained near
baseline in the current local seq-128 robustness run. That evidence is enough
to guide the next implementation branch, but it is not a substitute for the
full preregistered M-regime and A-vs-B sweeps.

The decision criteria for the program (keep / pivot / kill) are defined and
live in the source documents; they govern the sequence of events. There is no
schedule — there is only a sequence of events, each gated on the evidence from
the one before.

## What lives here (eventually)

- `research/` — specs, mechanism inventory, source integration, delegation plan.
- `src/grip/` — data generators, models, training, evaluation, analysis.
- `configs/`, `scripts/`, `tests/` — reproducibility.
- `runs/`, `reports/` — outputs (gitignored).

See `research/README.md` for the current contract order and the active
autonomous work queue.

## License

MIT — see [LICENSE](LICENSE). Permissive on purpose: the value here is the
mechanism idea and the experimental result, and adoption (grafting grip into
existing sparse-attention architectures) is the goal. Matches the MIT-licensed
codebases this project forks from.
