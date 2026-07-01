# Open-model mechanisms to borrow

Status: research collection as of 2026-07-01. This document records modern
open-model mechanisms that are relevant to Grip before we write sparse/grip
model code. It separates ingredients from full-scale training stacks.

## Ranking for toy Grip

1. OLMo Hybrid-style state/attention interleaving.
2. DeepSeek-style sparse attention selection seam.
3. Differential attention as a cheap signal-vs-noise ablation.
4. Qwen3-Next-style hybrid attention plus sparse MoE, as a heavier composite.
5. Mamba-2/RWKV-style recurrent state paths, if persistent state becomes the
   central question rather than attention selection.

## Mechanism map

| Family | Source | Mechanism | Reusable toy ingredient | Decision |
|---|---|---|---|---|
| DeepSeek-V2 | https://github.com/deepseek-ai/DeepSeek-V2 | Multi-head Latent Attention (MLA) for KV compression plus DeepSeekMoE. | Low-rank latent KV memory and top-k expert routing. | Borrow concepts later; not needed for first A-vs-B. |
| DeepSeek-V3 | https://github.com/deepseek-ai/DeepSeek-V3 | MLA plus DeepSeekMoE with auxiliary-loss-free balancing and MTP. | Sparse routing without an explicit load-balancing aux loss. | Defer large-scale training recipe. |
| DeepSeek-R1 | https://github.com/deepseek-ai/DeepSeek-R1 | RL/post-training pipeline on V3-base style model. | Training recipe only, not an attention mechanism. | Defer. |
| DeepSeek-V3.2-Exp | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp | DeepSeek Sparse Attention (DSA): indexer-driven sparse token selection for long context. | Direct analog for Grip-conditioned block/token selection. | Implement simplified local selector now; defer kernels. |
| Qwen3 | https://qwenlm.github.io/blog/qwen3/ | Dense and MoE model family. | Sparse expert routing baseline. | Defer unless module routing becomes central. |
| Qwen3-Next | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct and https://vllm.ai/blog/2025-09-11-qwen3-next | Hybrid attention using Gated DeltaNet plus full/gated attention, with high-sparsity MoE and MTP. | Mostly recurrent/linear layers with periodic exact attention refresh. | Use as a design reference now; defer full baseline reproduction. |
| OLMo Hybrid | https://allenai.org/blog/olmohybrid and https://huggingface.co/allenai/Olmo-Hybrid-7B | 3:1 pattern of Gated DeltaNet sublayers followed by one multi-head attention sublayer. | Clean toy recipe for state tracking plus periodic exact recall. | Primary add-state design reference; full imported baseline can wait. |
| Mamba / Mamba-2 | https://github.com/state-spaces/mamba | Selective state spaces and structured state space duality. | Non-attention persistent state backbone. | Defer until the question shifts to pure state tracking. |
| RWKV-7 | https://github.com/BlinkDL/RWKV-LM and https://wiki.rwkv.com/basic/architecture.html | Linear-time, constant-space recurrent language model family. | Online state update without KV cache. | Defer; less directly tied to attention selection. |
| Differential Transformer | https://arxiv.org/abs/2410.05258 | Differential attention subtracts one attention map from another to suppress noise. | Small ablation for signal cancellation under distractors. | Implement as a compact experimental module after B exists. |
| MoE routing | DeepSeek/Qwen sources above | Sparse selection over compute paths, not over context tokens. | Top-k router as a module-selection control. | Defer unless module routing is part of the question. |
| KV-cache selection | H2O, SnapKV, StreamingLLM, PyramidKV sources in `MECHANISM_INGREDIENTS.md` | Select, evict, or budget prior KV states at inference time. | Comparator for "which past states survive" without adding Grip state. | Include after local A-vs-B is wired, before claiming broad selection superiority. |

## What to copy versus what to imitate

Copy/adapt only small, inspectable pieces that are needed for instrumentation:

- block-index/top-k selector logic;
- local/window masks;
- simple low-rank KV projection;
- a small differential-attention module;
- optional recurrence block if a hybrid baseline is introduced.

Do not copy full model stacks for Stage 0:

- full DeepSeek MLA/MoE training stacks;
- production DSA kernels;
- full Qwen3-Next MoE/hybrid model;
- full RWKV or Mamba training recipes.
- KV-cache compression systems whose assumptions are inference-only unless
  the comparison is explicitly framed as inference cache selection.

The first result has to answer whether grip-conditioned selection beats an
equally supervised readable-grip control. Full frontier-model reproduction would
increase surface area without making that causal comparison sharper.

## Concrete borrowing plans

### DeepSeek / NSA / DSA

Use as the sparse-selection reference. The useful abstraction is:

```text
block_features -> indexer/scorer -> top_k block indices -> sparse attention
```

Grip modifies the scorer, not the whole attention stack:

```text
content_score(block) + lambda * grip_score(query_state, grip_summary(block))
```

For local training, implement this in ordinary PyTorch and expose trace fields:
candidate block scores, selected block indices, and read budget. For CUDA/cloud,
evaluate whether the FLA NSA `block_indices` seam can consume Grip's selected
blocks directly.

### OLMo Hybrid / Qwen3-Next

Use as the modern hybrid-state design reference. The useful idea is not the exact
large model, but the layer schedule:

```text
stateful update, stateful update, stateful update, exact/global attention
```

For Grip, this becomes the immediate add-state design reference:

```text
grip/state update layers plus periodic full or sparse attention refresh
```

The current derivative-supervised probe result says trajectory variables remain
near-baseline, so use this state/attention schedule as the default branch. Full
OLMo Hybrid or Qwen3-Next reproduction remains a later baseline.

### Differential attention

Use as a compact ablation against distractor sensitivity. A minimal version can
compute two attention maps and subtract/gate them before applying values. This
does not replace Grip, but it tests whether "cancel distractors" explains part
of the same gain. It is not part of the first A-vs-B gate, but it must run
before claiming broad distractor-robustness or selection superiority.

### MLA and MoE routing

Treat MLA as bounded memory compression and MoE as sparse compute routing. They
are useful analogies and later baselines, but they are orthogonal to the first
attention-selection result.

## Decision table

| Question | Best ingredient | Why |
|---|---|---|
| Can Grip improve which past blocks are read? | DSA/NSA-style top-k block selection | Same intervention surface as `SPEC-003` B. |
| Can a persistent state carry trajectory variables? | OLMo Hybrid / Qwen3-Next / Mamba-2 | State updates are explicit rather than inferred from hidden activations. |
| Can distractor cancellation mimic Grip's value? | Differential attention | Cheap falsification baseline. |
| Can generic sparse routing explain the gain? | MoE-style top-k routing | Tests selection over modules, not memory. |
| Can compressed KV memory explain the gain? | DeepSeek MLA | Tests bounded memory capacity rather than epistemic state. |

## Sources

Access date for web sources: 2026-07-01.

- DeepSeek-V2: https://github.com/deepseek-ai/DeepSeek-V2
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3
- DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1
- DeepSeek-V3.2-Exp: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek API release note: https://api-docs.deepseek.com/news/news250929
- Qwen3 blog: https://qwenlm.github.io/blog/qwen3/
- Qwen3-Next model card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- vLLM Qwen3-Next support note: https://vllm.ai/blog/2025-09-11-qwen3-next
- OLMo Hybrid blog: https://allenai.org/blog/olmohybrid
- OLMo Hybrid model card: https://huggingface.co/allenai/Olmo-Hybrid-7B
- Mamba repo: https://github.com/state-spaces/mamba
- RWKV repo: https://github.com/BlinkDL/RWKV-LM
- RWKV architecture wiki: https://wiki.rwkv.com/basic/architecture.html
- Differential Transformer: https://arxiv.org/abs/2410.05258
