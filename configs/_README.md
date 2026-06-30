# Configs

Reproducibility is mandatory: **same config + same seed => same run** (modulo
MPS nondeterminism, which is logged). One YAML per run template.

Naming: `<task>-<model>-<size>-<seq>.yaml`, e.g. `bayesian-dense-4M-512.yaml`.

Every config MUST record the **read budget** (top_k_blocks) for sparse variants
— the load-bearing A-vs-B comparison holds it equal.
