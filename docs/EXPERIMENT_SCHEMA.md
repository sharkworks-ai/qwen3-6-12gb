# Experiment Record Schema

Every experiment must have a stable `run_id` and machine-readable metadata.

Minimum logical schema:

```yaml
run_id: string
parent_run_id: string|null
created_at: ISO-8601
git_commit: sha
stage: baseline|sft|rlvr|profile|prune|recover|quantize|eval|release

hardware:
  gpu_name: string
  gpu_vram_mib: integer
  driver_version: string
  cuda_version: string
  cpu: string
  ram_gib: number

model:
  base_model: string
  base_revision: string
  checkpoint: string|null
  checkpoint_sha256: string|null

dataset:
  manifest: path
  manifest_sha256: string

compression:
  pruning_map: path|null
  quantization_map: path|null
  cpu_moe_layers: integer|null
  kv_k: string|null
  kv_v: string|null

runtime:
  engine: string
  commit: string|null
  command: string|null

metrics:
  peak_vram_mib: number|null
  prefill_tokens_per_second: number|null
  decode_tokens_per_second: number|null
  coding_score: number|null
  agent_score: number|null
  long_context_score: number|null
  tool_schema_validity: number|null

artifacts:
  - path: string
    sha256: string
```

## Storage

Use SQLite or DuckDB for searchable experiment summaries. Keep large per-example logs and trajectories as files referenced by the database.

## Promotion rule

A candidate is promoted only if it passes the hard release gates for its evaluation tier and either:

- improves at least one Pareto objective without materially worsening another; or
- is explicitly retained as a control/baseline.
