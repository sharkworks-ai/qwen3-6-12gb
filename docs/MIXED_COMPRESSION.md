# B / C mixed compression in the appliance

Open **B / C compression** in the single-container web UI. Select B or C,
set source/output/calibration paths, attention and shared precision, and sensitive
tensor rules. Validate and save the plan, then run the pipeline. Logs, stop,
VRAM telemetry and exact configuration use the existing Runs page. Use Resume
with the same configuration and output directory after interruption.

| Preset | Routed experts | Other projections | Recovery |
| --- | --- | --- | --- |
| B aggressive | GSQ-style 2-bit scalar reconstruction | 3/4-bit scalar | Optional QAT |
| C extreme | AYOT ternary reference reconstruction | Sensitive tensors GSQ 2/3-bit; attention/shared 3/4-bit | Required QAT and re-quantization |

The first two expert layers are example sensitivity overrides in C. They are
not measured sensitivity results. Replace them with rules derived from task
ablation. Unmatched overrides fail. Routers, normalization, embeddings, LM head,
biases and scalar state parameters remain full precision.

## Implementation boundaries

These are **local experimental reference implementations**. The scalar optimizer
learns discrete assignments through Gumbel-Softmax and per-group scales, using
RTN initialization and activation-second-moment-weighted reconstruction error.
It does not reproduce upstream GSQ's GPTQ initialization, full layer reconstruction,
Lion schedule, distributed optimiser or compressed-tensors export.

C adds source-model reasoning traces to ternary calibration. It is **ScaleQ-inspired**,
not a reproduction of CAT-Q's sliding-layer reconstruction or the ScaleQ paper.
The public BitTern tree inspected for this change contains CAT-Q inference/export
code, but no dedicated ScaleQ reconstruction entrypoint. No upstream benchmark
numbers can be attributed to this implementation.

Activation collection supports individual Linear projections and eager mm/bmm
operations using fused expert bank views. Opaque grouped expert kernels, offload
implementations that replace tensor storage, or missing expert coverage fail
instead of using fabricated statistics. Validate on the installed Transformers
revision before launching the full model.

This job loads the BF16 model with device_map=auto for calibration and uses
bounded row chunks during reconstruction. Two 5090s plus CPU RAM/offload may be
needed. It does not claim that full-model QAT fits those cards without additional
memory configuration. QAT uses the existing distributed FSDP Trainer path.

## Calibration and recovery

B consumes JSONL text or messages rows. C first needs **AYOT teacher traces**:
use the UI form to generate reasoning from the exact uncompressed source.
The prompt JSONL accepts prompt or messages fields without assistant answers.
Incomplete thinking traces fail. Source, configuration and corpus hashes protect
resume. Teacher provenance and corpus hashes are checked by C.

Recovery consumes a JSONL text/messages training dataset. This is selective
fake-quant supervised recovery, not teacher-logit distillation. The precision map
is shared by reconstruction and QAT. Recovery exports normal HF parameter names,
then runs reconstruction again. Keep calibration and benchmark datasets separate.

## Outputs and deployment

Each run stores plan.json, precision-map.json, activation moments, per-tensor
packed safetensors, progress.json with artifact hashes, mixed-manifest.json,
and research-hf. Recovery pipelines add qat/recovered and requantized outputs.

Ternary stores five base-3 codes per byte: **1.6 stored bits per code**, plus
group scales and metadata. INT2/3/4 use packed scalar codes. Packed-byte counts
cover quantized tensors only; unquantized tensors, KV, workspace and runtime
overheads must be added separately.

**There is no production kernel or GGUF/vLLM export for this mixed packed
format yet.** All manifests set runtime_ready=false. The research-hf checkpoint
uses BF16 storage. Passing it through the existing GGUF quantizer changes the
representation; that is a separate experiment, not lossless export of GSQ/C.

Existing Publish can upload result directories to GitHub/Hugging Face.
Do not label the artifacts as 12GB-ready. Coding, executable tool loops,
262144-token context, OxCoder comparison and peak VRAM release gates remain
unvalidated. The next required work is a production runtime/export adapter and
full-model GPU validation.

Config examples: configs/appliance/mixed-aggressive.json and mixed-extreme.json.
Equivalent launch: python -m appliance.quant.pipeline --config CONFIG.json.
