# Qwen3.6 12GB

Research and engineering project to build the strongest practical **Qwen/Qwen3.6-35B-A3B** derivative for autonomous coding and tool use on a **single 12 GB GPU**, while retaining the model's full **262,144-token native context window**.

## Primary goal

Produce a reproducible release that:

- serves a 262,144-token context window on a 12 GB GPU;
- maximizes coding and agentic task completion rather than benchmark-only quality;
- preserves all 40 transformer layers unless measured evidence supports depth pruning;
- uses workload-aware MoE expert pruning and mixed-precision quantization;
- keeps the full long-context KV history rather than relying on lossy KV eviction;
- is validated with complete tool-use and repository-editing trajectories;
- can be rebuilt automatically from recorded configs and experiment metadata.

## Working hypothesis

The most promising path is:

```text
Qwen3.6-35B-A3B
  -> agent/coding SFT
  -> executable tool-use / RLVR stage
  -> target-workload expert profiling
  -> structured expert pruning (~50-55% retained)
  -> recovery distillation
  -> mixed Q3/Q4 routed-expert quantization
  -> higher precision always-active path
  -> Q4 KV cache
  -> llama.cpp
  -> 262,144 tokens on 12 GB
```

A quality-first variant may keep Q4 experts and offload a small amount of MoE weight data to system RAM if that materially improves agent performance.

## Hard release constraints

| Constraint | Target |
|---|---:|
| Base model | Qwen/Qwen3.6-35B-A3B |
| GPU VRAM | 12 GB |
| Context | 262,144 tokens |
| Peak target VRAM | <= 11.5 GiB |
| Tool-call schema validity | >= 99% |
| Agent score retention | >= 95% vs tuned uncompressed checkpoint |
| Coding score retention | >= 95% vs tuned uncompressed checkpoint |
| Long-context retention | >= 95% vs tuned uncompressed checkpoint |
| Runtime | llama.cpp primary |
| KV cache | Q4 K/V baseline |
| Routed expert floor | ~3-bit mainline; ternary is experimental |

## OxCoder-9B external benchmark gate

The project is only considered worthwhile if the final 12 GB / 262K model also surpasses **OrionLLM/OxCoder-9B**:

- beat OxCoder on at least **6 of its 11 published benchmark rows**;
- beat it on at least **5 of the 8 coding + agentic rows**;
- beat it on at least one SWE-bench result;
- beat it on at least one Terminal-Bench 2.1 harness.

Comparisons must use matched published methodology where possible. See [docs/OXCODER_BASELINE.md](docs/OXCODER_BASELINE.md).

## Repository map

```text
configs/        Search, training, pruning, quantization and eval configs
docs/           Architecture, research plan and operating notes
src/qwen12g/    Experiment controller and project utilities
scripts/        Bootstrap and worker scripts
eval/           Agent, coding, long-context and general eval definitions
runs/           Local experiment outputs (gitignored)
```

The full implementation and research plan is in [docs/PLAN.md](docs/PLAN.md).

## Status

**Phase 0: repository and automation foundation.**

The repository is being prepared for a remote RTX 5090 worker. The 5090 is the development/training worker; the final 12 GB card remains the release target and must be used for final validation.

## Safety and credentials

Remote rented GPU workers must use temporary, least-privilege credentials. Do not forward personal SSH agents or broad GitHub/Hugging Face tokens into model-executed sandboxes.

## Dataset note

The proposed `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` dataset is useful for research, but its upstream provenance and licensing must be reviewed before any commercial redistribution or use. Keep dataset manifests and provenance with every training run.
