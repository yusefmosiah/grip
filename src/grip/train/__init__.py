"""Training loops, checkpointing, logging.

Design rules (from the operating model + M1 reconciliation):
  - Log every run to JSONL (machine-readable, append-only).
  - Microbatch + gradient accumulation; never large physical batches.
  - Fixed sequence lengths only (the MPS dynamic-shape leak — pytorch #181213).
  - fp16 + GradScaler OR fp32. Never bf16 on M1.
  - torch.compile OFF until a tiny reproducer proves it safe (pytorch #171764).
  - PYTORCH_ENABLE_MPS_FALLBACK OFF in dev (unsupported ops must fail loud).
  - Checkpoint/resume so a kill is harmless.
"""
