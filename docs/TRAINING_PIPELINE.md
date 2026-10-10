# Implemented appliance pipeline

## SFT / QLoRA

`appliance.stages.sft` uses Transformers + TRL + PEFT. The base checkpoint is loaded in NF4 by default, and LoRA targets both ordinary linear modules and the Qwen3.6 routed expert parameters (`mlp.experts.gate_up_proj` and `mlp.experts.down_proj`). The router itself is not targeted, so it remains frozen in the first SFT stage.

bitsandbytes 4-bit quantizes only `nn.Linear` layers. Qwen3.6's routed experts are fused 3-D parameters, so they stay BF16: of 35.1B parameters, 32.2B are experts (about 60 GiB), and a "4-bit" load is still about 62 GiB. A per-GPU replica therefore cannot fit a 32 GB card.

The wizard consequently profiles and prunes **before** SFT, then trains the pruned model as one process sharded across the selected GPUs (`model_parallel: true`, GPU-only `max_memory`). At 128 of 256 experts the BF16 model is about 33 GB, which fits 2x32 GB with room for adapters and activations. Training the pruned model also recovers quality lost to pruning. The DDP path (`torchrun`, one 4-bit replica per GPU) remains for models whose quantized replica fits a single card. If memory is tight, lower `lora_rank`, `expert_rank`, context length or `keep_experts`.

TRL consumes conversational and tool-calling datasets directly. `assistant_only_loss=true` is the default project policy.

## Expert profiling

The profiler registers forward hooks on Qwen3.6 `mlp.gate` routers. It records both selected-expert counts and normalized top-k routing mass for every layer on a representative agent/coding calibration set.

## Structured pruning

The pruner rewrites the safetensors checkpoint directly without loading the full 35B model. It ranks experts independently per layer from profiling data, slices the routed expert tensors and router matrix consistently, and updates `text_config.num_experts`.

Version 1 intentionally retains the same expert **count** in every layer because stock Transformers has one global `num_experts` field. Different layers may retain different expert identities. Variable counts per layer require a custom architecture/runtime and remain a later research branch.

After pruning, run SFT again against the pruned checkpoint for recovery.

## Calibration + GGUF quantization

The calibration stage materializes representative chat/tool text. The quantization stage uses the llama.cpp converter, optional `llama-imatrix`, and `llama-quantize` with configurable tensor regex overrides. The container builds llama.cpp with CUDA, so quantization is self-contained.

A typical first candidate is overall Q3_K_M, Q5 output/embedding tensors, and Q5 attention overrides. Exact tensor-pattern optimization is part of the automated search stage rather than hard-coded as a single supposedly optimal mix.
