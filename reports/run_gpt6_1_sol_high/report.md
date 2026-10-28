# bonsai-vk verification report

- build: **ok**
- **correctness (primary): 0.9958** (accepted)
  - tokenizer: 1.0000
  - dequant: 1.0000
  - logits: 0.9965
  - positions: 0.9789
  - greedy: 1.0000
- performance: geomean speedup vs llama.cpp Vulkan = 0.511

| test | agent t/s | reference t/s | speedup |
|---|---:|---:|---:|
| pp512 | 1292.96 | 2593.73 | 0.498 |
| pp2048 | 1274.20 | 2639.73 | 0.483 |
| pp4096 | 1249.07 | 2984.00 | 0.419 |
| pp8192 | 1202.89 | 2729.58 | 0.441 |
| tg128 @ d0 | 97.04 | 179.30 | 0.541 |
| tg128 @ d512 | 95.46 | 183.61 | 0.520 |
| tg128 @ d2048 | 91.14 | 177.67 | 0.513 |
| tg128 @ d4096 | 86.28 | 137.00 | 0.630 |
| tg128 @ d8192 | 77.07 | 131.84 | 0.585 |
- VRAM: agent peak 1720.0 MiB, reference 1967.1 MiB, gate PASS
