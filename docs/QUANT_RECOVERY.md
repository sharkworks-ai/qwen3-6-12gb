# Quantization recovery and QAT

The release search now treats quantization as an iterative compression/recovery loop.

## Policy

- >=98% coding/agent/long-context retention: accept PTQ result.
- 95-98%: build a teacher-failure replay mixture and run `post_quant_recovery`.
- <95% coding or agent retention: escalate to `qat_recovery`.
- Stop after two recovery rounds by default.

## Post-quant recovery

This reuses the project's SFT/QLoRA stage against a recovery mixture. The key extra data is teacher-success/student-failure replay gathered from the evaluation harness. This avoids loading a second 35B teacher during recovery.

## Selective QAT

QAT is performed after expert pruning, not on the original 35B BF16 model.

Fake quantization is registered as a PyTorch parametrization on selected parameters, so the forward pass sees quantization noise. This works with Qwen MoE 3-D expert parameters as well as ordinary 2-D weights.

When the precision map sits in a reconstruction bundle, QAT rounds each tensor onto that reconstruction's own grid: the per-group scales from its packed files times the runtime's integer levels (`use_reconstruction_grid`, default true). The reconstructed weights then pass through unchanged, so recovery starts from the reconstruction rather than re-rounding it onto a fresh min/max grid. Min/max scaling recomputes each group's step from its largest weights; when a GSQ group leaves the extreme levels unused (common for the 3/4-bit attention and shared tensors), that step is smaller and every weight is re-rounded. On a pruned Qwen3.6 aggressive reconstruction, this raised held-out loss from 1.51 to 2.48 before any training. Tensors without packed scales, or whose rows are not a multiple of the group size, keep min/max fake quantization.

By default (`trainable: "lora"`) the model is frozen and each quantized tensor trains a low-rank correction through the quantizer: the forward weight is `fake_quant(W + B @ A)`, with one adapter per expert for 3-D tensors (`lora_rank`, default 16; `lora_learning_rate`, default 1e-4). One process holds the pruned BF16 student sharded across the GPUs (`device_map="auto"`, GPU-only `max_memory`), with gradient checkpointing so each layer's fake-quantized weights are recomputed in backward. Removing the parametrization leaves the merged, fake-quantized weight that re-quantization starts from.

`trainable: "full"` keeps the previous FSDP full-parameter path. It needs weights, gradients and optimizer state for every parameter (roughly 8 bytes per parameter, about 140 GB for a 128-expert student), which does not fit 2x32 GB GPUs.

Modes:

- `q4_dense`
- `q3_moe`
- `q2_moe`
- `ternary_moe`

QAT output is never treated as the deployable artifact. The exact target quant backend (llama.cpp, TurboQuant, CAT-Q) is run again after QAT, then the candidate is fully reevaluated.

## Backend policy

- Standard Q3/Q4: QAT is optional and score-triggered.
- TurboQuant weight formats: use the same recovery loop when quality drops.
- TurboQuant KV-only compression: skip QAT because model weights are unchanged.
- CAT-Q: run native CAT-Q first; only escalate to QAT if agent/coding retention remains below threshold.

## Automatic Release Search state machine

The Release Search controller now exposes `recovery_action(...)`. After every quantized evaluation it:

1. compares candidate scores with the tuned reference;
2. accepts candidates retaining at least the configured threshold;
3. launches `post_quant_recovery` for moderate regressions;
4. launches `qat_recovery` for larger coding/agent regressions;
5. rejects candidates that exhaust the recovery-round budget.

A recovery job never updates the Pareto frontier directly. The caller must rerun the original quant backend and the same evaluation tier first. This prevents a high-precision recovery checkpoint from being compared against deployable quantized artifacts.
