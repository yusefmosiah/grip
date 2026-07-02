# Mechanism ingredients

Status: research collection as of 2026-07-01. This is a mise en place document:
collect the components before writing the first sparse/grip model. The
repo/package term is Grip.

## Ground rules

- The repo is the experimental harness, not a fork of any single model repo.
- Stage 0 must run on local M-series Mac hardware using ordinary PyTorch code.
- CUDA/Triton kernels are useful later, but they must not be required for the
  first correctness sweep.
- `SPEC-002` owns milestone order: M-legibility, M-probe, M-regime, M-avsb.
  Do not build Grip mechanisms before the legibility/probe gates have results.
- `SPEC-003` is the causal authority once A-vs-B is reached. The central
  comparison is still `grip-read (A)` vs `grip-select (B)` under matched
  params, matched FLOPs, matched read budget, and preregistered seeds.
- External code can be imported or adapted only when the license is compatible
  and the selection traces can be observed.

## Existing local ingredients

The repo already has the task and analysis foundation:

| Area | Local artifact | Role |
|---|---|---|
| Stream generator | `src/grip/data/streams.py` | Emits fixed-length Bayesian evidence streams with posterior, entropy, `d_conf`, `dd_conf`, source trust, and decisive positions. |
| T1 task | `src/grip/data/reversal.py` | Focused source-reliability reversal stream for the A-vs-B comparison. |
| Dense baseline | `src/grip/models/dense.py` | Full-attention backbone and posterior head for probe work. |
| Derivative probe | `src/grip/analysis/probe.py` | Frozen-backbone linear probes for level vs trajectory variables. |
| Bypass gate | `src/grip/analysis/bypass.py` | Raw-token leakage test before any model result is interpretable. |
| Metrics | `src/grip/eval/metrics.py` | Object metrics, reconstruction metrics, MI, and decisive-token recall with explicit position block ids. |
| Sweep preregistration | `src/grip/eval/sweep_plan.py` | Machine-readable `SPEC-003` sweep matrix and validation. |
| Stub: sparse models | `src/grip/models/sparse.py` | Placeholder for local/content-sparse/grip-readable/grip-select models. |
| Stub: trainer | `src/grip/train/run.py` | Must produce artifacts only, not winner claims. |
| Stub: scorer | `src/grip/eval/score.py` | Must own comparison and winner selection. |

## First model suite

These are the model variants to implement before adding the wider zoo:

| Variant | Mechanism | Why it is needed | Implementation posture |
|---|---|---|---|
| Dense/full | Standard causal self-attention | Upper reference and probe backbone. | Already present; keep as control. |
| Local-only | Sliding-window attention | Cheap floor; proves full context is needed. | Implement locally with PyTorch masks/SDPA. |
| Content-sparse | Local window plus top-K content blocks | Stock baseline that grip must beat. | Implement locally first; use NSA repos as references. |
| Generic-memory | Content sparse plus generic learned slots | Capacity confound for grip summaries. | Implement locally. |
| Grip-read (A) | Content selection, readable grip summaries, lambda=0 | Tests whether aux supervision alone reshapes representations. | Implement locally and instrument traces. |
| Grip-select (B) | Content score plus grip-conditioned selection term | The proposed mechanism. | Implement locally, then compare with NSA-style selectors. |
| Shuffle/wrong-sample grip | B with incorrect grip state | Causal-use control. | Implement as selector input perturbations. |
| Bottleneck-off | B with unconstrained grip | Leakage control. | Implement only after bottlenecked B works. |

## Attention and memory zoo

| Family | Primary sources | Practical source | License posture | Stage |
|---|---|---|---|---|
| Full attention / SDPA | PyTorch SDPA docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | PyTorch built-in | Project dependency | Stage 0 |
| Longformer | Paper: https://arxiv.org/abs/2004.05150; repo: https://github.com/allenai/longformer | AllenAI repo or HF implementation | Apache-2.0 | Early baseline |
| BigBird | Paper: https://arxiv.org/abs/2007.14062; repo: https://github.com/google-research/bigbird | Reference repo/HF implementation | Apache-2.0 | Early baseline |
| Native Sparse Attention (NSA) | Paper: https://arxiv.org/abs/2502.11089 | `lucidrains/native-sparse-attention-pytorch` and `fla-org/native-sparse-attention` | MIT | Core reference |
| DeepSeek Sparse Attention (DSA) | Repo: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp | Official inference/indexer reference | MIT | Core reference, not Stage 0 kernel |
| Performer | Paper: https://arxiv.org/abs/2009.14794; repo: https://github.com/lucidrains/performer-pytorch | lucidrains package | MIT | Early baseline |
| Linformer | Paper: https://arxiv.org/abs/2006.04768; repo: https://github.com/lucidrains/linformer | lucidrains package | MIT | Early baseline |
| Nystromformer | Paper: https://arxiv.org/abs/2102.03902; repo: https://github.com/mlpen/Nystromformer | Official repo/HF | Apache-2.0 | Early baseline |
| Transformer-XL | Paper: https://arxiv.org/abs/1901.02860; repo: https://github.com/kimiyoung/transformer-xl | Official repo or x-transformers recurrence | Apache-2.0 / MIT paths | Early baseline |
| Reformer | Paper: https://arxiv.org/abs/2001.04451; repo: https://github.com/lucidrains/reformer-pytorch | lucidrains package | MIT | Later suite |
| Routing Transformer | Paper: https://arxiv.org/abs/2003.05997; repo: https://github.com/lucidrains/routing-transformer | lucidrains package | MIT | Later suite |
| Long Range Arena | Paper: https://arxiv.org/abs/2011.04006; repo: https://github.com/google-research/long-range-arena | Benchmark reference | Apache-2.0 | Benchmark, not model |
| H2O | Paper: https://arxiv.org/abs/2306.14048; repo: https://github.com/FMInference/H2O | Heavy-hitter KV-cache eviction plus recent-token retention | Repo license must be checked before reuse | KV-cache comparator |
| SnapKV | Paper: https://arxiv.org/html/2404.14469v1; repo: https://github.com/FasterDecoding/SnapKV | Observation-window-driven KV-cache compression by selecting important clustered positions per head | Repo license must be checked before reuse | KV-cache comparator |
| StreamingLLM / attention sinks | Paper: https://openreview.net/forum?id=NG7sS51zVF; repo: https://github.com/mit-han-lab/streaming-llm | Retain attention sinks and recent tokens for streaming generation | Repo license must be checked before reuse | KV-cache comparator |
| PyramidKV | Paper: https://arxiv.org/html/2406.02069v1; repo: https://github.com/IsaacRe/PyramidKV | Layerwise dynamic KV-cache budget allocation | MIT reported by repo metadata; verify before reuse | Later KV-cache comparator |

## NSA / DSA selection seam

The best sparse-selection ingredients found in this research pass are:

| Source | What it gives us | Decision |
|---|---|---|
| `fla-org/native-sparse-attention` | Explicit `parallel_nsa_topk(...)` and `parallel_nsa(...)` functions that produce/consume block indices. Best kernel-level seam. | Adapt later for CUDA/cloud. |
| `lucidrains/native-sparse-attention-pytorch` | Importable `SparseAttention` module with block size and selected block controls. The selector uses top-k internally. | Best readable PyTorch reference. |
| `deepseek-ai/DeepSeek-V3.2-Exp` | Official DSA inference indexer behavior and model-side top-k token selection. | Reference only for Stage 0; do not require TileLang/FlashMLA locally. |
| `Open-Superintelligence-Lab/deepseek-sparse-attention-research` | Educational Lightning Indexer and top-k selector scaffolding. | Reference only, not primary. |

Grip's nearest intervention surface is the NSA/DSA selector: compute block
importance, choose top-K blocks, and expose selected block indices. The first
local implementation should copy the concept, not the CUDA kernel.

## Runtime split

| Runtime | Use for | Avoid for |
|---|---|---|
| Local MPS / CPU | Correctness, toy sweeps, small dense/local/content-sparse/grip variants. | Triton, FlashAttention, xFormers memory-efficient kernels, FlexAttention backward. |
| CUDA cloud | Longer contexts, NSA kernels, FlexAttention experiments, xFormers/FlashAttention comparison. | Defining the first scientific result before local correctness exists. |

PyTorch source currently indicates FlexAttention does not support backward on
MPS when inputs require gradients:
https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py.
Therefore local training should use PyTorch SDPA or ordinary tensor masks until
the experiment is stable.

## Gate-aware implementation order

1. Finish configs/artifact plumbing needed to run existing M-legibility and
   M-probe gates reproducibly.
2. Run M-legibility. If the bypass probe reads `d_conf` from raw tokens above
   the noise floor, harden the generator before model work.
3. Run M-probe. Current derivative-supervised probe evidence supports the
   absent-derivative branch, so the next design should add an explicit Grip
   state channel. Keep the selector-wire branch documented as the counterfactual
   if later probe evidence shows readable derivatives.
4. Implement local-only attention with trace output.
5. Implement content-sparse top-K block selection with trace output.
6. Run M-regime and prove content-sparse has a headroom gap below dense.
7. Add generic-memory slots.
8. Add grip summaries readable by attention, but keep selection content-only
   for variant A.
9. Add grip-conditioned selection for variant B.
10. Add shuffle/wrong-sample and bottleneck-off ablations.
11. Only then import/adapt wider zoo baselines and KV-cache comparators.

## Source ledger

Access date for web sources: 2026-07-01.

- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch FlexAttention: https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
- PyTorch MPS notes: https://docs.pytorch.org/docs/stable/notes/mps.html
- DeepSeek-V3.2-Exp: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- NSA paper: https://arxiv.org/abs/2502.11089
- lucidrains NSA: https://github.com/lucidrains/native-sparse-attention-pytorch
  (research pin from wave 1: `83fa271c21a0db35ef10d38d1d50aba8a09d3c69`)
- FLA NSA: https://github.com/fla-org/native-sparse-attention
  (research pin from wave 1: `bd67af59b90afa34b25f61d2922e612d10dba3bd`)
- DeepSeek-V3.2-Exp research pin from wave 1:
  `87e509a2e5a100d221c97df52c6e8be7835f0057`
- Longformer: https://arxiv.org/abs/2004.05150 and https://github.com/allenai/longformer
- BigBird: https://arxiv.org/abs/2007.14062 and https://github.com/google-research/bigbird
- Performer: https://arxiv.org/abs/2009.14794 and https://github.com/lucidrains/performer-pytorch
- Long Range Arena: https://arxiv.org/abs/2011.04006 and https://github.com/google-research/long-range-arena
- H2O: https://arxiv.org/abs/2306.14048 and https://github.com/FMInference/H2O
- SnapKV: https://arxiv.org/html/2404.14469v1 and https://github.com/FasterDecoding/SnapKV
- StreamingLLM: https://openreview.net/forum?id=NG7sS51zVF and https://github.com/mit-han-lab/streaming-llm
- PyramidKV: https://arxiv.org/html/2406.02069v1 and https://github.com/IsaacRe/PyramidKV
