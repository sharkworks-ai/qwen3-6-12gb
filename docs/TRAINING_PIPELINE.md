# Implemented appliance pipeline

## SFT / QLoRA

`appliance.stages.sft` uses Transformers + TRL + PEFT. The base checkpoint is loaded in NF4 by default, and LoRA targets both ordinary linear modules and the Qwen3.6 routed expert parameters (`mlp.experts.gate_up_proj` and `mlp.experts.down_proj`). The router itself is not targeted, so it remains frozen in the first SFT stage.

Two-GPU runs launch with `torchrun --nproc-per-node=2`. Each process loads a 4-bit copy on its local 5090 and DDP synchronizes adapter gradients. This is deliberately simple and robust for 2x32 GB. If adapter memory proves too high, lower `lora_rank`, `expert_rank`, or context length before moving to FSDP.

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
