# B / C mixed compression in the appliance

Open **B / C compression** in the single-container web UI. Select B or C,
set source/output/calibration paths, attention and shared precision, and sensitive
tensor rules. Validate and save the plan, then run the pipeline. Logs, stop,
VRAM telemetry and exact configuration use the existing Runs page. Select Resume
with the same configuration and output directory after interruption.

| Preset | Routed experts | Other projections | Recovery |
| --- | --- | --- | --- |
| B aggressive | Upstream GSQ 2-bit | Upstream GSQ 3/4-bit attention/shared | Optional QAT |
| C extreme | CAT-Q ternary grid with sliding-window adaptation | Sensitive GSQ 2/3-bit; attention/shared GSQ 3/4-bit | Required QAT and re-quantization |

The first two expert layers are example sensitivity overrides in C, not measured
sensitivity results. Replace them with rules derived from task ablation.
Unmatched overrides fail. Routers, shared-expert gates, normalization, embeddings,
LM head, biases and non-matrix state parameters remain full precision. The native
262144-token context configuration is retained; calibration length is separate.

## Pinned upstream components and training

The default `upstream_window` engine loads these exact upstream revisions:

| Source | Revision | Components used |
| --- | --- | --- |
| [GSQ](https://github.com/IST-DASLab/GSQ) | `03fc16484c369e3127225615d5e03e8d3a6043e3` | GPTQ and Gumbel INT2/3/4 quantizers |
| [BitTern CAT-Q](https://github.com/IntelChina-AI/BitTern/tree/main/projects/cat-q) | `5a8fcd4f7e0366554b732d300a37e6ea467c3c35` | Published ternary quantizer and scale/round factor parameterization |

Docker includes both checkouts and Lion 0.2.5. Jobs reject changed tracked source
or mismatched revisions, record package versions, and bind resume to provenance.
The earlier tensor-only engine remains selectable as `reference` for comparison.

GSQ initialization uses full activation covariance and the actual upstream GPTQ
algorithm. Fused expert banks use covariance pooled across expert views, not
independent Hessians for each expert. Eager mm/bmm/addmm capture supports individual
projections and fused expert-bank storage views. Missing coverage fails, with no
RTN or identity-Hessian fallback. Opaque grouped kernels are unsupported.

Reconstruction optimizes full decoder-block outputs against cached uncompressed
teacher outputs using Lion, cosine learning rates, and annealed Gumbel temperature
and logit strength. Student inputs propagate through already committed blocks.
B defaults to one block per window. C defaults to two overlapping blocks,
advancing one block at a time. Actual HF block arguments are cached so hybrid
linear/full-attention layers retain their individual call signatures.

C is **ScaleQ-inspired**, not an upstream ScaleQ reproduction. Public CAT-Q code
at the pinned revision is inference/export code without a training entrypoint.
Our training adaptation enables its factor gradients, softens rounding with a
straight-through estimator, and learns an independent low-rank correction for
each expert before quantization. The correction is merged into ternary weights;
there is no separate floating-point residual at inference. Defaults are rank 4
and alpha 4. No published ScaleQ benchmark numbers apply to this adapter.

## Web UI controls and resources

The JSON editor exposes `window_size`, `window_stride`, `window_devices`,
`checkpoint_steps`, `masks_lr`, `gptq_damp`, `gptq_block_size`, `catq_rank`, and
`catq_alpha`. Require stride <= width. Window blocks are placed round-robin on
the listed devices within one worker process. This is not GSQ's upstream
distributed launcher. Use `["cuda:0"]` for a single GPU. With B's default width
of one, only the first listed device is used during reconstruction.

The teacher checkpoint is loaded in CPU memory. Calibration moves one block
at a time to `device`; reconstruction holds only the active window on GPUs.
Provision CPU RAM for the complete BF16 source and disk for cached block inputs,
teacher outputs, full Hessians, packed weights and BF16 exports. These caches
can be large. QAT trains low-rank corrections through the quantizer on a frozen,
GPU-sharded student by default (see `docs/QUANT_RECOVERY.md`) and has a different
memory footprint. Full-model fit on two 5090s is unvalidated.

## Calibration, recovery and resume

B consumes JSONL text or messages rows. C first needs **AYOT teacher traces**:
use the UI form to generate reasoning from the exact uncompressed source.
Prompt JSONL accepts prompt or messages fields without assistant answers.
Incomplete thinking traces fail. C verifies teacher provenance and corpus hashes.
Keep calibration and benchmark datasets separate.

Recovery consumes a JSONL text/messages training dataset. It performs selective
fake-quant supervised recovery. Reconstruction and QAT share the precision map.
Recovery exports normal HF parameter names, then the same reconstruction engine
re-quantizes the result. Teacher provenance stays tied to the original source.

Window checkpoints include quantizer factors, low-rank corrections, optimizer
state and CPU/CUDA RNG state. Resume restores the active window and carries
overlap factors into the next. Source, configuration, upstream versions, teacher
caches, committed outputs and active checkpoint hashes are checked. Interrupted
teacher-cache collection restarts from scratch. Checkpoints are written at the
configured interval and at every window's final step.

## Outputs and validation

Each run stores `window-plan.json`, `precision-map.json`, `teacher-cache`, packed
safetensors, `window-progress.json`, `training-checkpoints`, `mixed-manifest.json`,
and `research-hf`. Recovery pipelines add `qat/recovered` and `requantized`.
Ternary stores five base-3 codes per byte: 1.6 stored bits per code plus scales.
INT2/3/4 use packed scalar codes. Packed bytes exclude unquantized tensors,
KV cache, workspace and runtime overhead.

A bounded eager runtime can now read this format directly. Use **12GB proof**
for comparisons and packed text generation. It replaces quantized projections
without loading their BF16 copies and decodes only active routed experts in row
chunks. See [proof-run and runtime details](PROOF_RUN.md).

There is no production kernel or GGUF/vLLM export for this mixed packed format.
Manifests set `runtime_ready=false`. `research-hf` uses BF16 storage for evaluation
and recovery. The existing GGUF quantizer changes the representation and is a
separate experiment. Existing Publish can upload result directories.

CPU tests exercise real pinned quantizer arithmetic/backward, packed-grid parity,
tiny standard and hybrid Qwen MoE reconstruction, HF export reload, and exact
interrupted-run resume. Only the upstream CUDA RNG calls are substituted in the
CPU harness. Production jobs require CUDA. Full-model GPU execution, coding,
executable tool loops, 262144-token context, OxCoder comparison and peak VRAM
release gates remain unvalidated. Artifacts are not yet 12GB-ready.

Config examples: `configs/appliance/mixed-aggressive.json` and
`configs/appliance/mixed-extreme.json`. Equivalent launch:
`python -m appliance.quant.pipeline --config CONFIG.json`.
