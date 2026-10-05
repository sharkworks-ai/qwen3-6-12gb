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

QAT is performed after expert pruning, not on the original 35B BF16 model. The pruned BF16 student is sharded across the two RTX 5090s with FSDP.

Fake quantization is registered as a PyTorch parametrization on selected parameters, so the forward pass sees quantization noise while the underlying master parameter remains trainable. This works with Qwen MoE 3-D expert parameters as well as ordinary 2-D weights.

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
