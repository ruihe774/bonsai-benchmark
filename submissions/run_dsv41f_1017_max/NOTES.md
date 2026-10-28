# bonsai-vk — implementation notes

Vulkan compute inference engine for `Ternary-Bonsai-2-4B-PTQ1_0.gguf`, written in
Zig 0.17.0 against hand-written Vulkan bindings (no third-party bindings or
libraries). Build: `zig build -Doptimize=ReleaseFast` → `zig-out/lib/libbonsai.so`,
which exports exactly the C ABI in `ABI.md`.

## Design

- **GGUF parsing** (`src/main.zig`): streaming reader over the file (pre-allocated
  buffers, `pread`); all metadata, tensor table, tokenizer dictionary and merges
  are read at runtime from the GGUF. No weights, tensors, token strings or
  results are embedded in the binary; the only embedded data is the compiled
  SPIR-V (14 compute shaders).
- **Weights**: the GGUF tensor-data region (~895 MiB) is uploaded once into a
  single device-local buffer in 32 MiB chunks. Every shader indexes this buffer
  at absolute element offsets passed via push constants, so no per-tensor
  descriptor sets, copies or re-packing are needed. PTQ1_0 tensors are decoded
  on the fly by the matvec shader (bit-exact decode verified against the
  reference dequantization); remaining f32 tensors are read directly.
- **Compute**: all model math runs in Vulkan compute shaders — embedding row
  decode + inverse Hadamard, RMSNorm, forward/inverse block-Hadamard (512-block
  Walsh–Hadamard with the GGUF sign vectors, explicit per-width sign offsets),
  ternary matvec, f32 matvec, conv1d+SiLU, L2 norm, gate/beta, gated delta-net
  recurrence, RoPE, f16 KV store, GQA causal attention (global-memory softmax),
  SwiGLU, residual elementwise ops, argmax.
- **Activation scratch**: one 1.29 MiB device-local buffer with fixed 256-byte
  aligned regions addressed by offsets; a single per-step command buffer with
  global memory barriers between dispatches (no render passes, no host sync
  inside the forward pass).
- **State**: f16 KV cache (8 full-attention layers × k/v), f32 GDN state
  (24 layers × 32×128×128), f32 conv history (24 × 3 × 8192). Allocated device
  local; sized by `n_ctx` at context creation (1 ≤ n_ctx ≤ 16384).
- **Host work only**: tokenization, GGUF I/O, command/descriptor orchestration
  and one logits readback per eval. No CPU fallback exists: if Vulkan is
  unavailable, `bonsai_model_open` reports an error and eval fails.
- **Tokenizer** (`src/tok.zig`): GPT-2 byte-level BPE with the `qwen35`
  pre-tokenizer, driven entirely by the GGUF dictionary/merges; Unicode
  L/M/N/space classification from a generated compact range table
  (`src/uni_tables.zig`), control/user-defined special tokens matched atomically.

## Model details derived during development

- Hybrid qwen35: 32 layers, GDN on all but every 4th layer (il%4==3 uses full
  attention). GDN: conv1d k=4 + SiLU, q/k = 16 heads × 128 (q scaled by 1/√128),
  v = 32 heads × 128 tiled from 16 k-heads, decay-first recurrence
  `S = S·exp(g) + k⊗(β(v − k·S_decayed))`, gated RMSNorm (×silu(z)), interleaved
  head permutation, then `T` + `ssm_out`.
- Full attention: q projection is **per-head interleaved** `[q(256)|gate(256)]×16`
  (not a chunked q/gate split); per-head q/k RMSNorm, partial RoPE (64 dims,
  θ=1e7, pairs i/i+32), causal GQA (4 kv heads), `attn·sigmoid(gate)`, then
  `T` + output projection.
- Hadamard: normalized Sylvester–Walsh blocks of 512; weights are stored
  transformed, activations get `T = H(S·x)` (signs before Hadamard) and the
  embedding/output use the inverse `S·H`. `ssm_alpha`/`ssm_beta` consume the
  untransformed normalized activation.
- GDN gate: `g = ssm_a·softplus(α + dt_bias)`, `β = sigmoid(b)`.

## Measured results

Correctness (vs the provided CPU golden logits / reference runner):

| test | result |
|---|---|
| PTQ1_0 dequant | bit-exact vs reference (4 tensors checked) |
| tokenizer, 15 visible vectors | 15/15 exact ids |
| tokenizer, fuzz vs `bonsai-ref tokenize` (Unicode, combining marks, emoji, code, whitespace, specials, random) | 92/92 exact |
| logits, prompt `[9419]` vs `oracle_logits_t1_cpu.bin` | cos 0.999993, KL 1e-4, top-1 and top-5 identical |
| logits, prompt `[248044, 248046]` vs `oracle_logits_a.bin` | cos 0.999965, KL 1e-4, top-1 and top-5 identical |
| greedy 8 tokens (vulkan reference, same prompt) | 8/8 identical |
| position handling | positions 0…16383 supported; multi-token eval matches incremental eval |

Performance (RADV, n_ctx=16384): ~46 tok/s prefill, ~35 tok/s decode (single-token
forward ≈ 22–29 ms including submit/fence and logits readback).

Memory at n_ctx=16384: peak VRAM ≈ 1.94 GiB (≈906 MiB weights + ≈565 MiB KV
cache/states/scratch); peak host RSS ≈ 118 MiB (weights are streamed, not mapped).

## Known limitations

- Prefill processes tokens one at a time (no batched/prefill GEMM), so prompt
  ingestion runs at decode speed rather than batched speed. Correctness is
  unaffected; this is the main remaining performance gap.
- KV cache is f16 (same as the reference Vulkan path), so logits differ from the
  f32 CPU golden in the last digits (cos ≈ 0.99997), not in the argmax.
- One GPU device is used; no multi-GPU split.
- n_ctx is capped at 16384 (per the task); n_batch is accepted but unused.
- Invalid UTF-8 in `bonsai_tokenize` is handled byte-wise rather than rejected.

## Source map

`src/main.zig` (GGUF parse, ABI, engine orchestration), `src/vk.zig` (Vulkan
bindings + helpers), `src/tok.zig` (BPE tokenizer), `src/uni_tables.zig`
(generated Unicode categories), `shaders/*.comp` (GLSL compute sources),
`src/spv/*.spv` (embedded compiled shaders), `tools/` (development-only
verification scripts, not needed by the build).
