# Quantization backends

The appliance treats quantization as a pluggable stage.

## `llama_cpp`

Production/default backend.

Supports:

- standard GGUF quant types;
- imatrix calibration;
- per-tensor `--tensor-type` overrides;
- Q4/Q5/Q8 KV types at runtime.

This remains the release baseline.

## `turboquant`

Experimental but directly useful for Qwen3.6.

Two separate features are exposed:

### TurboQuant weight formats

Examples:

- `TQ1_0`
- `TQ2_0`
- `TQ3_1S`
- `TQ3_4S`
- `TQ4_1S`

These require a TurboQuant-capable llama.cpp fork.

### TurboQuant KV cache

Runtime-only options such as:

- `turbo2`
- `turbo3`
- `turbo4`

The same weight model may be benchmarked with ordinary Q4/Q8 KV or TurboQuant KV.

The appliance records the required runtime in `quantization-result.json` so a
fork-dependent GGUF is never presented as an upstream llama.cpp artifact.

## `bittern_catq`

Experimental research backend for CAT-Q / 1.58-bit ternary PTQ.

BitTern officially publishes Qwen3 and Qwen3-MoE examples such as Qwen3-30B-A3B and
Qwen3-235B-A22B. At the time this backend was added, Qwen3.6 is not an explicitly
released upstream target.

Therefore:

1. the backend runs an architecture probe;
2. Qwen3.6 requires `allow_experimental_qwen36=true`;
3. the pinned BitTern revision's CAT-Q CLI must be explicitly configured;
4. agent-loop benchmarks are mandatory before a ternary artifact can be promoted.

This prevents the UI from implying unsupported Qwen3.6 compatibility.

## Search policy

The automated search should compare at least:

- pruned + Q4/Q3 standard GGUF;
- pruned + mixed standard GGUF;
- TurboQuant TQ2;
- TurboQuant TQ3 family;
- CAT-Q ternary experts when the Qwen3.6 adapter passes validation.

For each candidate record:

- final file size;
- actual peak 12 GB VRAM;
- 262K context success;
- prefill/decode speed;
- coding score;
- agent-loop score;
- OxCoder benchmark wins;
- required runtime/fork.

A smaller artifact is not a winner if it damages agent reliability.
