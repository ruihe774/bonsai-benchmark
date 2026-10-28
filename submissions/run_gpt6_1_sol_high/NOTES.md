# bonsai-vk: design and verification

## Build and runtime

`zig build -Doptimize=ReleaseFast` produces `zig-out/lib/libbonsai.so`, exporting
the ABI in `examples/abi/bonsai.h`. The implementation is Zig 0.17 plus GLSL
compute shaders. The generated Zig Vulkan declarations are checked in; no
`@cImport`, dependency fetch, reference engine, or prebuilt engine is involved.
The build compiles and embeds SPIR-V and does not read `data/`, `docs/`, or
`ref/`. A fresh source-only build in a separate directory without those three
directories passed. Runtime dependencies are only the Vulkan loader and libc.
`dlopen` and model initialization were tested with SDK/loader/ICD environment
variables removed.

Only the GGUF path supplied by the caller is opened as model data. Models are
mmap'd, parsed with checked sizes/offsets, and uploaded in their original packed
representation. Contexts retain their model so releasing a model handle before
its contexts does not destroy live GPU resources. Diagnostics are thread-local.

## GPU execution

Every forward-pass operation, including state clearing and greedy argmax, is
Vulkan compute. CPU code handles GGUF/tokenizer parsing, tokenization, resource
orchestration, uploads, and result I/O. CPU tensor dequantization is used only
by the required introspection ABI, never by inference. No CPU fallback exists.

The device requests Vulkan 1.4 and queries/enables the compute Roadmap 2026
features from DEVICE.md, including the KHR cooperative matrix, maintenance,
shader-control, untyped-pointer, robustness2, clock, derivatives and pipeline
binary extensions. Missing required features cause initialization to fail.
Shaders actually use device-address buffers, scalar layout, f32/f16 arithmetic,
8/16-bit storage, fixed 32-lane subgroups, and KHR cooperative matrices.

### Model structure

* 32 residual blocks, embedding width 2560, SwiGLU intermediate width 9216.
* 24 gated DeltaNet blocks, with four-tap causal depthwise convolution,
  16 key heads and 32 value heads of width 128, f32 recurrent state.
* Eight gated full-attention blocks with 16 query heads, four KV heads, head
  dimension 256, and 64 rotary dimensions. KV storage is f16.
* The tied embedding/output matrix and dense weights use PTQ1_0. Explicit-sign
  normalized Hadamard512 transforms come from the GGUF metadata.

PTQ1_0 has 128 elements per 28-byte block: 26 permuted packed ternary bytes and
one fp16 scale. Introspection returns raw transformed-domain tensor values,
bit-exactly. Embedding lookup applies the inverse Hadamard transform; dense
projections apply the forward transform to their inputs.

DeltaNet value heads have raw order `group*16+key_head`. After gated RMS
normalization, outputs are permuted to adjacent grouped heads before the output
projection. This ordering was recovered from GGUF metadata and documented
reference tensor traces, not binary inspection.

### Execution strategy

* Decode uses packed ternary matvec shaders sharing activations across four
  output rows per subgroup; the vocabulary projection uses eight rows.
* Prefill uses layer-major batches of up to 512 tokens. A single reusable
  45 MiB f16 scratch area expands **one projection at a time** on the GPU;
  there is never a dequantized copy of the whole model. A 64-token/64-output
  cooperative GEMM tile reuses each weight fragment across four accumulators.
* DeltaNet recurrence runs in shared memory for prefill, with subgroup partial
  reductions, and uses a parallel f32 state update for decode.
* Prefill attention uses cooperative QK/PV products and online f32 softmax.
  Decode splits attention values across up to 128 context partitions before a
  deterministic f32 reduction. Cache allocations are padded to 64 positions to
  make tail tiles safe even at non-aligned context sizes.
* All state and temporary allocations are bounded by the context/batch sizes.
  Greedy argmax breaks ties in favor of the lowest token id.

## Correctness measurements

Against the documented CPU reference (not merely the Vulkan reference):

* Tokenizer: all 15 provided vectors and 2,033 additional randomized
  multilingual/combining-mark/whitespace/code/special-token strings matched
  exactly.
* Tensor dequantization: 1,230 sampled slices, 152,832 f32 values, across all
  426 tensors matched the safetensors development copy bit-for-bit.
* Diverse logits suite: 30 prompts, including multilingual text, source code,
  chat markers, arithmetic and corpus excerpts up to 2,096 tokens; prompt logits
  plus four teacher-forced decode steps each: **149/150 top-1 matches**, mean
  KL **0.0000938 nats**, maximum KL 0.00113, mean cosine **0.999979**,
  minimum cosine 0.999836.
* 8,192-token prefix plus 64 teacher-forced steps: **100% top-1 agreement**,
  mean KL **0.00000572**, mean cosine **0.999964**.
* Eight-token free-running greedy smoke test matched the CPU reference exactly.
* Incremental evaluation, deterministic resets, capacity/argument errors,
  thread-local diagnostics, retained model lifetime, and context sizes
  1/17/65/513 were exercised through the actual C ABI.
* Debug parser fuzzing: 2,001 mutated/truncated/synthetic GGUF inputs returned
  normally without a panic or out-of-bounds failure.
* Vulkan validation layers reported no validation errors during complete
  inference and ABI tests (only the expected RADV conformance warning).

These are development tests, not an official hidden-grader score. Both suites
were rerun successfully on the final 512-token internal-batch configuration.

## Performance and memory

Measured through the ABI, n_ctx=16,384, three repetitions, 128 generated tokens
with EOS ignored; medians. The comparison runner uses its documented Vulkan
backend and identical tokenized prefixes. All numbers below use the final
512-token internal-batch configuration.

| Prompt tokens | bonsai pp tok/s | reference pp tok/s | bonsai tg tok/s | reference tg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 25.5 | 225.5 | 97.4 | 179.5 |
| 512 | 1209.2 | 2632.0 | 96.2 | 184.7 |
| 2048 | 1255.8 | 2663.8 | 91.3 | 177.8 |
| 4096 | 1249.0 | 3011.0 | 86.5 | 137.5 |
| 8192 | 1201.4 | 2752.9 | 77.4 | 131.8 |

The geometric mean ratio over the four pp lengths and five tg depths was
0.504x. Prompt length one is a short-prefix startup measurement, not a bulk pp
case. The depth-zero tg case is approximated by a one-token initial prefix.

Observed peak VRAM was **1,720 MiB**, versus **3,041 MiB** for the reference;
the acceptance bound from that reference peak is **4,561 MiB**. Peak host RSS
was about **1,028 MiB**, versus **2,038 MiB** reference (and an allowed bound of
4,086 MiB). Both measured memory gates pass. The packed model remains resident
on the GPU throughout inference.

## Limitations

* Inference is specialized to this model's dimensions, quantization and
  architecture; incompatible models are rejected instead of misinterpreted.
* One sequence per context; no sampling beyond greedy generation or chat
  template application. An empty generation prompt requires prior evaluated
  state.
* Context length 16,384 is tested. Larger requests are permitted up to the
  training length, but are untested and can fail device allocation.
* f16 KV/activation conversion and reduction order differ from the CPU oracle;
  logits are tolerance-equivalent, not bit-identical. Tensor introspection is
  bit-identical.
* The implementation is slower than the highly optimized reference. Prefill
  projection bandwidth and the sequential DeltaNet recurrence remain the main
  optimization opportunities.
