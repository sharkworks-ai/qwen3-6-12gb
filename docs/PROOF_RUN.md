# 12GB proof run and packed inference

Open **Reports & tests** in the container web UI. The default single-GPU run creates
about 160M parameters from scratch with a byte tokenizer. It uses the target
family's hybrid linear/full attention, routed expert banks and shared experts.
It retains a 262144-token configuration while testing short sequences.

The sequence is:

1. Generate seeded source weights, tokenizer and synthetic fixtures.
2. Train the source for a short bootstrap stage.
3. Record held-out baseline loss.
4. Run B reconstruction, interrupt after a durable optimizer checkpoint and restart.
5. Evaluate the initial BF16 export, run selective QAT, then re-quantize.
6. Load packed artifacts directly, measure held-out loss and compare logits with the BF16 export.
7. Repeat stages 4–6 for C.

All stages run sequentially in separate processes. Large models from earlier
stages do not remain GPU-resident. Stage logs, CUDA/device memory traces, duration,
checkpoints and output hashes are stored under the output directory. Resume
verifies configuration and artifact hashes and skips completed stages. A failed
stage is visible in the report and can be resumed. Source/training datasets and
checkpoints remain under the mounted data directory, not Git.

Select a physical card with `cuda_devices`, such as `"0"` or `"1"`; the worker
uses logical `cuda:0`. Start with the defaults. Peak allocated GPU memory is
reported separately from sampled total device memory. The proof limit is 11.5 GiB.
There is no measured pass when GPU telemetry is missing. The default model size
is a starting configuration, not a promise of fit on every 12GB card.

## What this proves

This is a synthetic process test. The short bootstrap is not language-model
pretraining, and its handwritten reasoning examples are not AYOT teacher traces.
C accepts these only in explicit `proof_mode` on a model carrying the miniature
proof marker. Real C runs still require verified AYOT provenance. The report marks
AYOT, coding quality and full-context quality as untested.

The report compares initial and recovered loss, baseline logit error, packed/BF16
export parity, resident tensor bytes, artifact sizes, per-stage times and VRAM.
The planned process restart exercises saved optimizer/RNG checkpoints. Exact
interrupted-versus-uninterrupted packed hashes are checked by the existing small
MoE integration tests. The proof's pass is a process/resource gate and is never
release approval.

## Packed eager runtime

`appliance.runtime.packed.load_packed(BUNDLE, device="cuda:0")` builds the HF
architecture on meta, replaces quantized Linear modules and fused Qwen expert
banks, and loads only unquantized parameters from the BF16 research shards.
It reads INT2/3/4 bit codes and five ternary codes per byte directly. Each projection
decodes bounded output-row chunks. Each routed expert is decoded only when the
current batch selects it; there is no persistent full BF16 expert bank or cache.
Signed GSQ scales are preserved. Generation uses the HF model's standard cache.

The runtime imports PyTorch, Transformers, Accelerate and safetensors, not the
training framework or upstream reconstruction code. New bundles hash both packed
and research files. Older window artifacts remain readable using their block
export hashes, but lack complete outer/tokenizer integrity coverage.

This is an eager correctness and memory adapter. It is not a fused low-bit CUDA
kernel, GGUF format or vLLM plugin. Decode and expert loops can be slow, especially
on full-model prefills. Non-quantized weights, activations, decoding scratch and
KV/linear state still count toward VRAM. KV remains the HF runtime's native dtype;
this does not implement quantized full-history KV. Manifests retain
`runtime_ready=false` until production performance and release gates pass.

## Full-model validation

The adjacent form runs the real merged source through B and C with recovery,
held-out baseline comparison, packed runtime evaluation and repeated context
needle checks. Baseline defaults to GPU/CPU offload across the two 5090s; edit
`max_memory` for your host. Packed runtime uses `runtime_device` on one GPU.

Use separate calibration, AYOT, recovery and held-out corpora. The workflow binds
resume to their hashes and the source checkpoint. Long-context tests use SDPA
and check prompt plus output <=262144. Needle retrieval is only a diagnostic;
it does not replace repository-scale code reasoning or complete tool loops.
The default matrix reaches a 250000-token prompt, reserves 8192 output tokens,
and repeats each length five times. It can take substantial time and may OOM.

Coding and tool-use benchmark commands are not executed inside this container
with its publishing credentials. Run them in isolated task harnesses. Import an
optional JSON file using `benchmark_results`, with both `aggressive` and `extreme`
objects containing:

- `artifact_manifest_sha256`: hash of that final mixed manifest.
- `harness_revisions`: pinned harness/version metadata.
- `scores`: all 11 benchmark keys from `appliance.eval.oxcoder_gate.BASELINE`,
  each on a 0–100 scale.

The report applies the existing OxCoder gates and keeps release approval false.
It cannot certify imported results or infer missing coding/tool quality tests.
To add benchmark results after a completed run, use the separate **Benchmark
comparison** form. It binds scores to the existing final artifact without
re-running training or changing the completed workflow identity.

Open the report by entering its output directory on the proof page. Equivalent
CLI commands are `python -m appliance.proof.job --config CONFIG.json` and
`python -m appliance.proof.validation --config CONFIG.json`.

## 4GB NVIDIA laptop test

In **New run**, choose **Test the workflow on one GPU**, select GPU 0, and
choose **Laptop** under **Small test model size**. This creates a roughly 100M
parameter hybrid MoE with five layers, 64-token examples and two calibration
records. Set the memory pass/fail target to **3.0 GiB** on a 4GB display GPU.
Use 5 initial updates, 2 weight-fitting updates and 2 recovery updates for the
first run. A memory target is a report gate, not an allocation cap. Fit still
needs to be measured on the actual GPU.

Proof arithmetic is selected once and recorded in the run configuration:
BF16 on CUDA devices with native support, FP32 on older devices such as the
T1200. FP32 uses more memory and can run slowly. The same precision is used
for bootstrap, reconstruction, recovery and packed/export comparison. It does
not change the stored quantization bit widths. Full-model workflows retain
their existing BF16 requirements.

From the repository checkout, start the UI locally:

```bash
export QWEN12G_WEB_TOKEN="$(openssl rand -hex 24)"
export QWEN12G_PORT="127.0.0.1:8080"
export QWEN12G_CUDA_ARCHITECTURES="75"
export QWEN12G_BUILD_JOBS="2"
docker compose -f appliance/compose.yml up -d --build
printf 'Web UI login token: %s\n' "$QWEN12G_WEB_TOKEN"
```

Open `http://localhost:8080`. Keep the token locally. Architecture 75 is for
the Turing laptop; rebuild with the default architecture list for another
GPU host. Compilation is limited to two jobs by default to reduce host RAM
pressure. A complete image build can still need substantial disk space.
