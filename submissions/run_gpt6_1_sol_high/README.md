# bonsai-vk

A Zig 0.17 Vulkan 1.4 inference library for the supplied
Ternary-Bonsai-2-4B PTQ1_0 GGUF.

## Build

Requires Zig 0.17.0, `glslc` from Vulkan SDK 1.4.363.0, and the system Vulkan
loader/driver. No package downloads are needed.

```sh
zig build -Doptimize=ReleaseFast
```

The result is `zig-out/lib/libbonsai.so`. The SDK loader search path is included
in the library's runtime search path; callers do not need to set environment
variables. GLSL is compiled to SPIR-V during the build and embedded in the
library. Neither model data nor the reference tools are needed at build time.

## Interface

Use `examples/abi/bonsai.h`; the semantics are documented in `ABI.md`.

1. Open the GGUF with `bonsai_model_open`.
2. Tokenize UTF-8 with `bonsai_tokenize`, or supply existing token ids.
3. Create a context with `bonsai_context_create` (16,384 positions supported).
4. Append tokens with `bonsai_eval`, or use `bonsai_generate_greedy`.
5. Reset a context to reuse it; close contexts before closing their model.

Errors are negative return values with a thread-local `bonsai_last_error`.
The library follows the single-calling-thread contract. An empty generation
prompt requires previously evaluated context state.

The architecture is intentionally specialized to this model. Unsupported
architectures, incompatible tensor shapes, missing required Vulkan features,
and allocation failures return errors. There is no CPU inference fallback.

## Implementation

* Runtime GGUF parsing, exact PTQ1_0/F32 introspection, and metadata-driven
  GPT-2 byte BPE with the Qwen35 Unicode pre-tokenizer.
* Packed weights remain resident in device-local Vulkan memory.
* GPU embedding decode, explicit-sign Hadamard transforms, normalization,
  gated DeltaNet, causal convolution, partial RoPE, attention, SwiGLU,
  projections, residuals, state clearing, and greedy argmax.
* Prefill processes up to 512 tokens per layer-major batch, using KHR
  cooperative matrices and a reusable f16 scratch area for **one projection**.
* Online-softmax cooperative attention for prefill, and split-context
  attention reductions for decode. KV storage is f16; recurrent state and
  activation accumulation are f32.

See `NOTES.md` for measurements and limitations. Development checks under
`tools/` use the documented reference protocol only; the library itself never
executes helper processes or opens other weight files.
