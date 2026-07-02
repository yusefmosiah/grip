# SPEC-003 Amendment 001: add-state branch and regime conditioning

Date: 2026-07-01

This amendment is prospective for A-vs-B work after the current M-regime gates.
It does not reinterpret prior M-regime records.

## 1. Add-State Branch Marker

The current local probe evidence supports the explicit add-state branch:
derivative-supervised dense backbones keep level probes readable while
`d_conf`/`dd_conf` probes remain near baseline in the current seq-128
robustness run.

Therefore, until stronger contrary evidence exists, `grip-read (A)` and
`grip-select (B)` should both use an explicit Grip state module. Variant A may
read/reconstruct that state but must not use it for selection. Variant B uses
the same state in selection.

This is an amendment marker for the post-probe add-state edit already reflected
in `SPEC-003-ablations-and-sweeps.md`.

## 2. Regime-Search Conditioning

Any future A-vs-B claim is conditioned on the fact that the headroom regime was
searched for first. The claim must be phrased as:

> In a regime constructed to have content-sparse headroom below dense,
> grip-conditioned selection improves over grip-readable selection at matched
> compute/read budget.

Reports must include the regime-search path alongside the A-vs-B result:

- which M-regime cells were tried;
- which cells pivoted or were blocked;
- which cell first satisfied the pre-declared M-regime aggregate rule;
- whether the final A-vs-B result is still within that selected regime.

## 3. Terminology

Use "pre-declared" for rules introduced during this program. Reserve
"preregistered" for rules fixed before the relevant data existed. Historical
decision records are not rewritten, but errata should clarify any older
overclaim.

## Spec Freeze

This amendment was written against these spec blobs:

- `SPEC-000-derivative-probe.md`: `8d88bfc3821f0e5f4cf54ce92ad622011daa2d62`
- `SPEC-001-tasks-and-data.md`: `2aab6a0078d2922087fcf5c57456d065d012aef9`
- `SPEC-002-eval-protocol.md`: `5658bc4327c74b913aa5d8983fa1a4140499f326`
- `SPEC-002-AMENDMENT-001.md`: `37cbd50a59e2c79206dd2519ccaa5ed4ebb12b48`
- `SPEC-003-ablations-and-sweeps.md`: `b0e6f1331e87e4550c656e4204dc96ac79bb0f8d`
