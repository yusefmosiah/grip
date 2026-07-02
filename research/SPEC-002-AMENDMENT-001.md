# SPEC-002-AMENDMENT-001: Trustworthy gates amendment

Date: 2026-07-02
Status: active prospective amendment to `SPEC-002-eval-protocol.md`

This amendment supersedes the corresponding gate rules in SPEC-002 for all
runs started after this amendment is committed. Earlier runs remain historical
evidence only and must not be cited as decision-grade M-regime, probe, or A-vs-B
results unless they satisfy this amendment.

Source spec blob hashes at drafting:

| Spec | Git blob |
|---|---|
| `SPEC-000-derivative-probe.md` | `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62` |
| `SPEC-001-tasks-and-data.md` | `2aab6a0078d2922087fcf5c57456d065d012aef9` |
| `SPEC-002-eval-protocol.md` | `5658bc4327c74b913aa5d8983fa1a4140499f326` |
| `SPEC-003-ablations-and-sweeps.md` | `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d` |

## 1. Noise-floor calibration

M-noise-floor is no longer an "identical seed versus identical seed" check.
Each calibration pair must use the same resolved task, model, sparse-attention,
training, and evaluation configuration, but the left and right runs must differ
in seed and generated stream order. Calibration seeds must be disjoint from the
decision-run seeds used by M-regime or A-vs-B cells.

For each metric, the calibration artifact records every paired delta and sets
the metric threshold from the observed cross-seed spread:

```text
threshold(metric) = max(abs(delta(metric)) for paired calibration runs)
```

The calibration is invalid if every absolute delta for a metric is at or below
the declared numeric zero tolerance. Invalid calibration blocks any downstream
keep/pivot/kill decision that depends on that metric. It must fail loudly; it
must not silently fall back to an epsilon threshold.

Each calibration artifact must include:

- resolved config provenance for both sides of every pair;
- left/right seed pairs and decision-seed exclusions;
- per-metric deltas and thresholds;
- the numeric zero tolerance used for invalidity checks;
- a stable payload hash over the fields used for compatibility checks.

## 2. Decision-run compatibility

Any scorer that loads a noise-floor artifact must compare the artifact's
resolved config payload against each decision run before computing status.
Compatibility must cover at least task name, sequence length, batch shape,
model family, model dimensions, sparse-attention mode, content budget, training
step budget, optimizer family, evaluation batch size, and evaluation seed
policy.

If any checked field differs, the scorer must return a blocked result naming
the mismatched fields. A mismatched noise floor must never be used to authorize
a keep, pivot, kill, or aggregate status.

## 3. Minimum valid M-regime cell

M-regime outputs are split into `smoke` and `valid` tiers. A smoke tier is
allowed for debugging and plumbing, but it is unciteable by aggregate reports
and cannot authorize Grip A-vs-B work.

A valid M-regime cell must satisfy all of the following minimums unless a later
amendment raises them:

| Requirement | Minimum |
|---|---:|
| decision seeds per comparison cell | 8 |
| sequence length | 512 |
| evaluation batch size | 8 |
| training batch size | 8 |
| training steps | 1000 |
| calibration pairs for the relevant noise floor | 8 |

The scorer must refuse to emit `keep`, `pivot`, or `kill` for a decision-grade
M-regime cell unless the cell satisfies the valid tier and has a compatible,
non-degenerate noise floor. Smoke artifacts must carry `tier: "smoke"` and an
`unciteable: true` marker.

## 4. Aggregate keep criterion

A single seed or single favorable operating point is not decision-grade
evidence. For each declared task and comparison family, the aggregate status is
`keep` only when all conditions hold:

- every included cell is `valid`;
- every included cell uses a compatible, non-degenerate noise floor;
- at least 75% of decision seeds individually exceed the metric-specific
  noise-floor threshold in the declared direction;
- the aggregate mean delta exceeds the metric-specific noise-floor threshold;
- the declared read-budget curve is reported rather than a selected point.

If any condition is missing, the aggregate status is `blocked`. If the
conditions are present and the thresholds are not met, the status is `pivot`.

## 5. Multi-task authorization

Program-level authorization is never `any(task keeps)`. Each experiment must
declare its required task set before scoring. Program-level authorization for
Grip A-vs-B is true only when every required task has a valid aggregate `keep`.

Task-specific keeps may still be reported as local evidence, but they do not
authorize broader mechanism work unless the required task set is satisfied.

## 6. Probe thresholds and bypass controls

The derivative-probe gate from SPEC-000 keeps its qualitative interpretation,
but reports must include these fixed thresholds:

| Probe target | Decision threshold |
|---|---:|
| `topmass`, `entropy`, `source_trust` level controls | R2 >= 0.50 |
| `d_conf`, `dd_conf` derivative readability | R2 >= 0.20 |
| raw-token bypass to derivative targets | R2 <= 0.05 |
| raw-token bypass to answer labels | accuracy <= 1.6x chance |

A derivative-supervised robustness run is decision-useful only if it reports
the same threshold table and marks whether the derivative targets were included
in backbone auxiliary supervision.

## 7. Terminology

Use "declared" or "pre-declared" for rules introduced or changed during the
current research program. Reserve "preregistered" for rules that were fixed
before the relevant data was generated. Historical documents may keep their
original wording, but new decision records must identify which rules were
prospective for the run being reported.
