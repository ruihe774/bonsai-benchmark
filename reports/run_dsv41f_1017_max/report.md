# bonsai-vk verification report

- build: **ok**
- **correctness (primary): 0.4870** (below 0.90 gate)
  - tokenizer: 0.0000
  - dequant: 1.0000
  - logits: 0.0023
  - positions: 0.4297
  - greedy: 0.9062
- performance: geomean speedup vs llama.cpp Vulkan = 0.000 (illustrative subset)

| test | agent t/s | reference t/s | speedup |
|---|---:|---:|---:|
| pp512 | 44.91 | 2596.97 | 0.017 |
| pp2048 | 42.12 | 2621.74 | 0.016 |
| pp4096 | 39.05 | 2965.53 | 0.013 |
| tg128 @ d0 | 33.70 | 179.60 | 0.188 |
| tg128 @ d512 | 32.72 | 183.14 | 0.179 |
| tg128 @ d2048 | 30.40 | 177.62 | 0.171 |

- perf mode: illustrative subset (correctness took 1069s > 900s); the full 5-rep pp/tg run was skipped.
- VRAM: agent peak 4078.4 MiB, reference 1967.1 MiB, gate FAIL
