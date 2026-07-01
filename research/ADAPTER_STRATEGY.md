# Adapter strategy

Status: research collection as of 2026-07-01. This document decides how external
attention/model code should enter the Grip repo after the preregistered gates
allow sparse/grip model work.

## Policy

1. Keep `/Users/wiz/grip` as the harness and artifact authority.
2. Obey `SPEC-002` order: M-legibility, M-probe, M-regime, then M-avsb.
3. Prefer local PyTorch implementations for Stage 0 so traces, masks, selected
   blocks, and grip summaries are fully observable.
4. Import external packages when they are stable, licensed compatibly, and used
   as baselines rather than modified science mechanisms.
5. Copy/adapt small MIT/Apache-2.0 components only when instrumentation requires
   changing internals.
6. Pin source SHAs for copied/adapted code and record them in a source ledger.
7. Do not require CUDA, Triton, FlashAttention, or xFormers for the first
   correctness result.

## Local adapter contract

Sparse/grip model variants should expose a common output surface:

```python
{
    "lm_logits": Tensor,           # next-token/object logits
    "posterior": Tensor | None,    # if the model has a posterior head
    "hidden": Tensor,              # token hidden states for probes
    "selected_blocks": Tensor,     # block ids selected per query/block
    "selection_scores": Tensor,    # pre-top-k scores, masked where invalid
    "grip_state": Tensor | None,   # per-block or per-token grip state
    "grip_recon": dict | None,     # posterior/entropy/d_conf/dd_conf/source_trust heads
}
```

The add-state branch has two separable contracts:

```text
tokens -> grip state update -> grip summaries/recon heads
tokens -> block summaries -> candidate scores -> top-k block ids -> attention
```

The scorer remains separable from the attention application:

```text
tokens -> block summaries -> candidate scores -> top-k block ids -> attention
```

For A/B, the explicit grip-state producer is shared. Only whether that state
enters the candidate score changes:

```text
score = content_score + lambda * grip_score
```

Variant A must still build/read grip summaries, but its selector must ignore
them (`lambda=0` and no hidden path around the selector).

## Copy/import/adapt decisions

| Source | Decision | Rationale |
|---|---|---|
| PyTorch SDPA | Use directly. | Local correctness path, maintained, no vendoring. |
| PyTorch FlexAttention | Defer training to CUDA/cloud until upstream MPS backward support changes. | Current PyTorch source indicates MPS backward is unsupported; local smoke tests are only useful for forward/no-grad/debug paths. |
| xFormers / FlashAttention / Triton | Defer to CUDA/cloud. | Kernel/runtime dependencies are not local M1 training foundations. |
| `lucidrains/native-sparse-attention-pytorch` | Import or read as reference first; copy only if selector instrumentation needs it. | MIT, readable top-k selection, practical PyTorch API. |
| `fla-org/native-sparse-attention` | Adapt later for CUDA. | Best explicit `block_indices` seam, but Triton/CUDA constraints are not Stage 0. |
| `deepseek-ai/DeepSeek-V3.2-Exp` | Reference only. | Official DSA indexer behavior; public repo points to external kernel stacks for high-performance kernels. |
| Longformer/BigBird/Performer/Linformer/Nystromformer/Transformer-XL | Use as baselines after A-vs-B is wired. | They broaden the suite but do not answer the first causal question. |
| H2O/SnapKV/StreamingLLM/PyramidKV | Use as KV-cache selection comparators after local sparse/grip traces exist. | They test retention/eviction of prior states rather than Grip-conditioned training-time block selection. |
| OLMo Hybrid/Qwen3-Next/Mamba/RWKV | Use as design references now; import full baselines later. | The current probe result favors an explicit state channel, but full hybrid/recurrent stacks are not needed for the first A-vs-B implementation. |

## Backend lanes

### Lane 1: local correctness

- Device: CPU or MPS.
- Dependencies: PyTorch only where possible.
- Attention: SDPA or explicit tensor operations.
- Outputs: full traces and JSON artifacts.
- Goal: make every `SPEC-003` variant runnable at small scale.

### Lane 2: baseline package imports

- Device: CPU/MPS when feasible, CUDA where required by package.
- Dependencies: pinned package versions.
- Goal: compare against established mechanisms without modifying their internals.

### Lane 3: cloud kernels

- Device: CUDA.
- Dependencies: Triton, FlashAttention, xFormers, FLA NSA, and FlexAttention
  only after a CUDA smoke test proves the exact training mask works.
- Goal: scale context length or test kernel-level NSA integration after the
  local A-vs-B result exists.

## Gate-aware work queue

1. Add config files for the existing dense/probe/bypass/sweep contracts.
2. Fill `src/grip/train/run.py` enough to emit artifacts for M-legibility and
   M-probe without self-reporting winners.
3. Fill `src/grip/eval/score.py` enough to own comparison artifacts.
4. Run M-legibility and M-probe. Current robustness evidence supports the added
   state-channel branch; keep the wire-over-existing-trajectory branch as the
   counterfactual for later contrary evidence.
5. Implement the shared sparse output contract in `src/grip/models/sparse.py`,
   including `lm_logits`, `selected_blocks`, `selection_scores`, and
   `grip_recon` keys for `d_conf` and `dd_conf`.
6. Add trace tests for `selected_blocks`, `selection_scores`, and corrected
   decisive-token recall block-id semantics.
7. Implement local-only attention.
8. Implement content-sparse top-K selection.
9. Run M-regime to show content-sparse leaves headroom below dense.
10. Implement generic-memory, grip-read A, grip-select B, and ablation controls.

This order keeps the science falsifiable: once model code exists, scoring and
artifact boundaries already prevent the trainer from self-reporting a win.

## Licensing and attribution checklist

For every external code path:

- record repo URL, license, pinned commit or package version;
- record whether the code is imported, copied, or only referenced;
- keep copied code isolated under a clear module path;
- preserve license headers for copied files;
- add a test that proves the adapter emits the common trace contract.

Known pins from the first research wave:

- `deepseek-ai/DeepSeek-V3.2-Exp`: `87e509a2e5a100d221c97df52c6e8be7835f0057`
- `lucidrains/native-sparse-attention-pytorch`: `83fa271c21a0db35ef10d38d1d50aba8a09d3c69`
- `fla-org/native-sparse-attention`: `bd67af59b90afa34b25f61d2922e612d10dba3bd`

## Minimum acceptance for an imported baseline

An imported/adapted model is not part of the suite until it can:

- run from the same config/artifact path as local models;
- report matched parameter count and read budget where applicable;
- emit or reconstruct selected/read positions where applicable;
- run through `eval/score.py`;
- be excluded cleanly on unsupported hardware with an explicit skip reason.
