# Configs

Reproducibility is mandatory: **same config + same seed => same run** (modulo
MPS nondeterminism, which is logged). This directory currently records the
future YAML-template contract; the trainer accepts JSON config paths today, but
executable YAML templates and a YAML loader are not wired yet.

Future template naming: `<task>-<model>-<size>-<seq>.yaml`, e.g.
`bayesian-dense-4M-512.yaml`.

Every sparse-variant config MUST record the **read budget** (`top_k_blocks`);
the load-bearing A-vs-B comparison holds it equal.
