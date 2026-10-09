# Guided automated runs

The container opens the wizard at `/`. Advanced stage pages remain available.

1. Choose a miniature proof, training plus compression, or compression of an existing checkpoint.
2. Select exposed GPU indices and the final 12GB memory target. Baseline GPU/CPU allowances are separate.
3. Choose the local model and mounted JSONL datasets. Real checkpoints must retain native 262144 context. Held-out data cannot reuse an input file or its exact contents.
4. Choose B, C or both, training/reconstruction/QAT step budgets, and optional usage-based expert pruning. C uses CAT-Q from the pinned BitTern source, not an independent ScaleQ implementation.
5. Choose prompt length, output headroom, repetitions, and a configured benchmark suite.
6. Review the generated plan and launch one job.

The run performs SFT/QLoRA and adapter merge when requested, optional expert profiling/pruning, teacher reasoning generation for C, baseline diagnostics, selected reconstruction/QAT/re-quantization pipelines, packed held-out diagnostics, repeated context/memory checks, optional isolated published suites, and a candidate report. B and C use the eager packed runtime. KV quantization, production llama.cpp conversion and published agent harness implementations are not added by this wizard.

The proof builds synthetic data and always compares B and C. It cannot demonstrate real coding quality. Real-model diagnostic mode measures held-out loss and needle retrieval. It cannot claim a published benchmark win. Candidate recommendation requires measured resource/retrieval checks and the configured OxCoder benchmark gates. Release approval remains false even when a candidate meets these screening gates. Reference-retention and real full-context agent loops still need qualification.

## Data

Model checkpoints and input files must be under `QWEN12G_DATA_ROOT`, normally `/data`. The wizard consumes mounted files; model and dataset downloads are not automatic. Training and recovery accept text or conversational messages. Calibration and held-out diagnostics require `text` JSONL records. Teacher prompts contain `prompt` or user `messages`, without assistant answers. The service generates reasoning traces and their provenance manifest.

Artifacts are saved under `/data/artifacts/wizard/<name>`. `plan.json`, `input-manifest.json`, `proof-progress.json`, stage logs/checkpoints and `report.json` make the run reproducible. Every completed stage is hashed, including its nested model artifacts. Resume with the same name, files, code version and answers. It skips verified completed stages and retries the failed stage. Trainer checkpoints are used only when present. Stop uses the existing process-group cancellation. The run page refreshes while running and links to review/resume and the report. GPU launch checks reject missing or already-used devices.

A CLI run uses the same answers:

```json
{
  "answers": {
    "goal": "proof",
    "name": "first-proof",
    "cuda_devices": "0",
    "train_steps": 20,
    "quant_steps": 10,
    "qat_steps": 10,
    "vram_limit_gib": 11.5,
    "resume": false
  }
}
```

Run `python -m appliance.wizard.job --config /data/wizard-config.json`. The web runner applies `CUDA_VISIBLE_DEVICES`; direct CLI users must select the same physical GPUs themselves.

## Disk budget

A full-model run writes several BF16 copies of the model, so by default the wizard
deletes outputs that later stages no longer read, once those stages have succeeded:

- the merged SFT model, after pruning succeeds (training runs with pruning only);
- each variant's first-pass research export, first-pass packed bundle, window
  checkpoints and QAT directory, after that variant's re-quantization succeeds and
  before the next variant starts. The `requantized/` bundle is kept.

QAT keeps only its latest trainer checkpoint. Released paths are listed under
`released` in `proof-progress.json` and the stage hashes are re-recorded, so resume
still skips those stages. Set `"keep_intermediates": true` to keep everything.

`"release_source_after_merge": true` (training runs only, off by default) also
deletes the base checkpoint after merge, including the Hugging Face cache blobs a
snapshot links to. Resume reuses the source hashes recorded in `input-manifest.json`;
a new run needs the base model downloaded again.

## Automatic published benchmarks

Set `QWEN12G_BENCHMARK_PROFILES` to an operator-owned JSON file. The wizard lists its profile labels. Web users cannot submit a service URL, executable or shell command.

```json
{
  "coding-agent-suite": {
    "label": "Pinned coding and agent suite",
    "url": "https://your-harness.example/run",
    "revision": "your-pinned-harness-revision",
    "timeout_seconds": 86400,
    "token_env": "BENCHMARK_SERVICE_TOKEN"
  }
}
```

The isolated service must already exist and support the following contract. It runs model-generated code/tools outside the training container, owns its sandbox resource limits and networking, and can read the bundles through an approved shared artifact mount. It must support the `qwen12g_packed_eager` runtime. The adapter does not create a service or make unsupported harnesses compatible.

The container POSTs a synchronous request with `schema_version: 1`, a stable idempotent `request_id`, pinned `harness_revision`, `runtime`, and `candidates`. Each candidate carries `bundle` and `artifact_manifest_sha256`. Retries use the same request ID. Redirects are not followed. Service credentials come from the named environment variable and never appear in wizard answers or evidence.

The response maps each selected variant to:

```json
{
  "aggressive": {
    "artifact_manifest_sha256": "actual-manifest-sha256",
    "harness_revisions": {
      "profile": "your-pinned-harness-revision",
      "swe_bench_verified": "actual-harness-revision"
    },
    "scores": {
      "terminal_bench_terminus2": 0,
      "terminal_bench_claude_code": 0,
      "swe_bench_verified": 0,
      "swe_bench_pro": 0,
      "nl2repo": 0,
      "hle_no_tools": 0,
      "hle_with_tools": 0,
      "gpqa_diamond": 0,
      "mcp_atlas": 0,
      "browsecomp": 0,
      "claweval": 0
    }
  }
}
```

Include every selected variant and all eleven scores on a 0–100 scale. Artifact hashes and profile revisions must match. Placeholder numbers above illustrate the schema and are not results. Invalid evidence fails the stage. Evidence and comparison results are stored automatically. No manual results import is needed for a configured service. Without a service, diagnostics mode completes and reports that coding/tool quality is unproven.
