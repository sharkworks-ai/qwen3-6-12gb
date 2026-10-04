# Full Plan: Qwen3.6-35B-A3B on 12 GB with 262K Context

## 1. Objective

Build the strongest practical derivative of `Qwen/Qwen3.6-35B-A3B` for:

- autonomous coding;
- multi-tool agent workflows;
- repository-scale reasoning;
- shell and test-driven repair loops;
- full native 262,144-token context;
- deployment on a single 12 GB NVIDIA GPU.

The target is not merely to make the model load. The release candidate must preserve useful long-context reasoning and complete iterative tool-use loops under the 12 GB constraint.

## 2. Non-negotiable release requirements

| Requirement | Gate |
|---|---:|
| Native context window | 262,144 tokens |
| Release GPU | single 12 GB NVIDIA GPU |
| Peak VRAM target | <= 11.5 GiB |
| Tool-call schema validity | >= 99% |
| Agent success retention | >= 95% vs tuned uncompressed checkpoint |
| Coding score retention | >= 95% vs tuned uncompressed checkpoint |
| Long-context score retention | >= 95% vs tuned uncompressed checkpoint |
| Repeated full-context stability | 5/5 runs without OOM |
| Runtime | llama.cpp primary |
| Full-history cache | preferred; no lossy KV eviction in mainline |
| Depth pruning | last resort only |
| Ternary routed experts | research branch only until agent-loop quality is proven |

A practical full-context test must reserve output space. A 262,144-token context includes generated tokens, so the release suite should include prompts around 245K-250K tokens with 8K-16K available for reasoning and output.

## 3. Why this model

Qwen3.6-35B-A3B is well suited to the target because it combines:

- about 35B stored parameters;
- about 3B active parameters per token;
- 40 layers;
- 256 routed experts per layer, top-8 active routing;
- a shared expert;
- a hybrid architecture with 30 Gated DeltaNet / linear-attention layers and 10 full-attention layers;
- only 2 KV heads in the full-attention path;
- native 262,144-token context.

The hybrid architecture sharply reduces long-context KV growth compared with a conventional 40-layer transformer.

## 4. Core technical strategy

The mainline path is:

```text
Base Qwen3.6-35B-A3B
  -> baseline evaluation
  -> agent/coding SFT
  -> executable tool-use / RLVR stage
  -> workload-specific expert profiling
  -> structured expert pruning
  -> recovery distillation
  -> mixed-precision quantization
  -> Q4 KV
  -> llama.cpp deployment
  -> 262K validation on actual 12 GB GPU
```

The working hypothesis is that **moderate expert pruning plus 3-4 bit retained experts** will preserve more agent capability than keeping all experts and forcing the whole model into extreme 1-2 bit precision.

## 5. Target release variants

### 5.1 GPU-resident variant

Goal: all required model weights plus 262K KV/state fit in 12 GB.

Initial target:

- all 40 layers;
- early layers retain most or all experts;
- roughly 50-55% of routed experts retained overall;
- routed experts mostly Q3 with sensitive groups at Q4;
- shared expert Q5/Q6;
- attention and Gated DeltaNet Q5/Q6;
- router Q8 or FP16;
- norms FP16;
- embeddings/output Q5/Q6;
- KV Q4 K + Q4 V;
- text-only deployment.

Target model-weight budget: approximately 9.0-9.5 GiB, subject to measured runtime overhead.

### 5.2 Quality-first 12 GB variant

Goal: maximum coding/agent quality while still using the 12 GB GPU.

Allow:

- Q4 routed experts;
- a small amount of CPU MoE offload;
- otherwise the same 262K and quality gates.

This variant wins if a small system-RAM penalty yields materially better task completion.

## 6. What not to do first

Do not begin with:

- global IQ1/IQ2 quantization;
- uniform expert pruning;
- broad depth pruning;
- lossy KV eviction;
- training most examples at 262K;
- ternary conversion of all routed experts;
- router fine-tuning at the same learning rate as the rest of the model.

Each of those can save memory, but each can damage exactly the multi-step behavior this project is trying to preserve.

## 7. Phase 0: repository and reproducibility foundation

Deliverables:

- deterministic Python environment definition;
- CUDA / llama.cpp build notes for RTX 5090 and release cards;
- experiment config schema;
- run directory format;
- SQLite or DuckDB experiment database;
- structured benchmark result schema;
- dataset manifest format;
- hardware probe;
- VRAM sampler;
- command runner;
- checkpoint and artifact naming convention;
- bootstrap script for a fresh rented GPU host.

Every run must record:

```text
run id
git commit
base model revision
dataset manifest + hashes
training config
pruning map
quantization map
runtime commit/config
hardware details
CUDA/driver versions
benchmark results
VRAM trace
speed results
long-context results
agent trajectories
artifact hashes
```

## 8. Phase 1: untouched baselines

Create three baselines.

### B0: high-quality reference

Run the stock model at the highest practical precision on development hardware.

Purpose:

- establish reference coding score;
- establish reference agent score;
- establish long-context behavior;
- record router statistics before specialization.

### B1: existing 12 GB-style deployment

Use an existing GGUF configuration plus CPU-MoE offload and Q4 KV.

Purpose:

- prove 262K execution path;
- measure release-runtime behavior;
- establish speed and VRAM reference.

### B2: aggressive low-bit control

Use an existing full-model IQ2-ish quant.

Purpose:

- provide a control showing what is gained or lost by selective pruning/quantization.

Do not use B2 as the presumed final design.

## 9. Phase 2: evaluation harness before training

The evaluation harness must be built before model modification.

### 9.1 Coding

Use a mix of:

- HumanEval+/MBPP+ for fast screening;
- LiveCodeBench-style tasks;
- repository repair tasks;
- refactoring tasks;
- multi-file code-understanding tasks;
- test-driven bug fixing;
- build-system and dependency errors.

### 9.2 Agentic/tool use

Provide tools such as:

- `read_file`
- `search_files`
- `list_directory`
- `apply_patch`
- `run_command`
- `run_tests`
- `git_diff`

Score the final repository state and test results, not only the model's prose.

Include:

- multi-turn tool loops;
- bad-tool traps;
- malformed tool arguments;
- unavailable tools;
- stale observations;
- recovery after failed commands;
- cases where no tool should be called;
- repeated-command loop detection.

### 9.3 Long context

Evaluate at:

- 32K
- 64K
- 128K
- ~200K
- 245K-250K prompt length inside the 262K window.

Tests:

- multi-needle retrieval;
- RULER-like tasks;
- repository-scale symbol tracing;
- requirements placed far from implementation details;
- conflicting evidence across distant context regions;
- long tool histories;
- source + logs + documentation reasoning.

### 9.4 General retention

Keep a smaller suite for:

- reasoning;
- instruction following;
- basic factual competence;
- shell/config/networking knowledge relevant to agents.

## 10. Phase 3: dataset construction

Primary source candidate:

`r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`

Use the canonical records and deduplicate aggressively.

Proposed token-weighted mixture:

- 35% coding and repository tasks;
- 30% agent/tool trajectories;
- 15% long-context coding/repository tasks;
- 10% reasoning;
- 10% general capability retention.

The dataset should be augmented because long-context and realistic agent trajectories are not sufficiently represented by the source dataset alone.

### 10.1 Generated repository tasks

Build deterministic tasks from permissively licensed repositories:

```text
inspect repo
-> locate relevant code
-> edit
-> run test/build
-> parse failure
-> edit again
-> rerun
-> finish only when objective is satisfied
```

Store:

- repository revision;
- task prompt;
- allowed tools;
- expected tests;
- ground-truth patch when available;
- verifier script.

### 10.2 Long-context curriculum

Main training should stay relatively short.

Suggested distribution:

- majority: 4K-16K;
- substantial minority: 16K-64K;
- smaller set: 64K-128K;
- small anti-forgetting set: 128K-200K;
- 262K used mainly for evaluation unless long-context regression is detected.

### 10.3 Licensing

Track:

- source dataset;
- upstream license;
- teacher provenance;
- repository license;
- generated derivative status;
- commercial/research-use constraints.

Do not assume the proposed distillation dataset is suitable for commercial redistribution.

## 11. Phase 4: SFT

Use LoRA/QLoRA first.

Recommended initial policy:

- router frozen;
- norms frozen;
- embeddings frozen;
- adapt attention/GDN path;
- adapt shared expert;
- adapt routed-expert projections selectively;
- train using assistant/tool-call tokens only where appropriate.

Possible adapter strategy:

- always-active path: rank 16-32;
- routed experts: rank 4-16;
- prioritize experts seen frequently on agent/coding workloads.

Avoid independent high-rank adapters on every expert if adapter memory explodes.

### 11.1 Router stage

After initial SFT:

- optionally unfreeze router;
- use ~5-10x lower LR than other adapted weights;
- monitor expert entropy, load balance and collapse;
- stop if routing becomes sharply concentrated without task gains.

## 12. Phase 5: executable tool-use / RLVR stage

Use deterministic rewards where possible.

Primary rewards:

- tests pass;
- repository state is correct;
- requested change is complete;
- tool calls are valid;
- final answer matches actual state.

Secondary penalties:

- invalid JSON/tool arguments;
- repeated identical failed commands;
- fabricated tool results;
- runaway loops.

Do not strongly reward short trajectories. Hard tasks often require exploration.

Keep this stage reproducible with fixed sandbox images and verifier scripts.

## 13. Phase 6: expert profiling

Profile the tuned model, not only the base model.

Record per layer and per expert:

- selection count;
- mean routing probability;
- activation norm;
- task category;
- token position;
- coding frequency;
- tool-use frequency;
- long-context frequency;
- general-retention frequency.

Use millions of representative tokens if practical.

The pruning metric should combine:

- usage;
- routing mass;
- task-specific importance;
- ablation sensitivity;
- diversity contribution.

## 14. Phase 7: structured expert-pruning search

Do not prune every layer equally.

Candidate family:

| Candidate | L0-7 | L8-23 | L24-39 | Approx intent |
|---|---:|---:|---:|---|
| P0 | 256 | 256 | 256 | reference |
| P1 | 192 | 192 | 192 | moderate |
| P2 | 128 | 128 | 128 | 50% uniform |
| P3 | 256 | 128 | 96 | preferred adaptive |
| P4 | 256 | 96 | 96 | more aggressive |
| P5 | learned per-layer | learned | learned | search-generated |

P3 is the initial preferred research point.

Search should consider different retention counts per layer based on routing entropy and task importance.

If llama.cpp requires fixed expert counts, implement or maintain a small runtime patch for variable per-layer counts rather than forcing a worse pruning scheme.

## 15. Phase 8: recovery distillation

After pruning:

1. load tuned unpruned checkpoint as teacher;
2. load pruned student;
3. run SFT/recovery on the target workload;
4. include teacher/student KL or logit matching;
5. initially train router and adapters;
6. then allow a short low-LR broader recovery stage.

Evaluate after every recovery checkpoint and stop early if the Pareto score stops improving.

## 16. Phase 9: mixed-precision quantization search

Quantize using workload-sensitive calibration data.

Main search dimensions:

- routed experts: Q4, Q3, mixed Q3/Q4;
- shared expert: Q4-Q6;
- attention/GDN: Q4-Q6;
- router: Q8/FP16;
- output head: Q4-Q6;
- embeddings: Q4-Q6;
- norms: FP16;
- KV: Q4/Q4 baseline.

Suggested initial candidate:

```text
important routed experts    Q4
normal routed experts       Q3
shared expert               Q5
attention                   Q5/Q6
Gated DeltaNet              Q5/Q6
router                      Q8 or FP16
norms                       FP16
embeddings/output           Q5
KV K/V                      Q4/Q4
```

Do not use perplexity alone to rank quantization candidates. Full agent trajectories are mandatory.

## 17. Ternary / CAT-Q branch

Keep a separate experimental branch for BitTern/CAT-Q.

Test:

- ternary only on less-sensitive surviving routed experts;
- Q3/Q4 on high-importance experts;
- higher precision always-active path;
- packed ternary kernel performance.

A ternary candidate only graduates if it passes complete run-test-fix-retest agent loops at or above the mainline candidate.

Storage wins alone are insufficient.

## 18. Depth pruning branch

Use only if the expert-pruning + quantization path still misses the target.

Preserve architectural rhythm.

Test removal of one full four-layer group at a time rather than arbitrary single layers.

Rank groups using block influence and target-workload ablation.

Recovery-tune after each removal.

Stop if reasoning, tool planning or long-context integration drops disproportionately.

## 19. KV and long-context strategy

Mainline:

- full 262,144 context;
- Q4 K;
- Q4 V;
- no token eviction;
- no SnapKV-style pruning unless required.

Reason: the hybrid architecture already keeps full-context KV/state relatively cheap, and full history is highly valuable to coding agents.

Optional research:

- Q5/Q4 asymmetric KV;
- TurboQuant-style 4/3-bit cache;
- more aggressive cache compression only if model-weight budget cannot otherwise be met.

## 20. Runtime

Primary release runtime: llama.cpp.

Required capabilities:

- Qwen3.6 hybrid architecture;
- mixed GGUF quantization;
- Q4 KV;
- selective CPU-MoE offload;
- prefix cache;
- CUDA support for release GPU;
- Blackwell support for 5090 development worker.

Baseline launch shape:

```bash
llama-server \
  -m model.gguf \
  -ngl 99 \
  -c 262144 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

For quality-first variants, sweep `--n-cpu-moe`.

Benchmark Flash Attention on/off rather than assuming it is always better under tight memory.

## 21. Prompt-prefix caching

Treat prefix caching as a first-class feature for agent use.

Agent loops repeatedly reuse:

- system prompt;
- tool definitions;
- repository context;
- prior tool history.

Measure:

- cache hit rate;
- prefilling saved;
- VRAM impact;
- latency across repeated tool loops.

## 22. MTP / speculative decoding

Keep MTP as an optional speed branch.

Do not make MTP required for release.

Only enable it if:

- 262K remains stable;
- VRAM still meets the hard limit;
- agent/coding quality is unchanged;
- generation latency improves.

## 23. Automated experiment controller

The controller should:

1. generate candidate config;
2. provision/verify environment;
3. train or recover if needed;
4. prune;
5. quantize;
6. convert to GGUF;
7. run cheap screening;
8. run medium evaluation;
9. run expensive long-context evaluation only for survivors;
10. record metrics;
11. update the Pareto frontier;
12. propose the next candidate.

Use Optuna multi-objective search or a small custom evolutionary controller.

Do not collapse everything into one scalar score.

Objectives:

maximize:

- coding success;
- agent task completion;
- long-context accuracy;
- decode speed.

minimize:

- VRAM;
- CPU offload;
- prefill latency;
- quality regression.

## 24. Staged evaluation funnel

### Tier 0: smoke

- model loads;
- basic chat;
- one tool call;
- short code generation;
- no NaNs/crashes.

### Tier 1: cheap

- small coding set;
- tool-schema validity;
- 8K agent loops;
- short perplexity/reference metrics.

Kill clearly bad candidates.

### Tier 2: normal

- larger coding set;
- repository tasks;
- 32K/64K context;
- multi-tool loops.

### Tier 3: expensive

- 128K/200K long context;
- harder repository tasks;
- repeated agent loops;
- speed/VRAM profiling.

### Tier 4: finalist

- 245K-250K prompt;
- output headroom retained;
- full-context tool loop;
- repeated OOM/stability test;
- actual 12 GB GPU validation.

## 25. Remote RTX 5090 worker

The 5090 is the training and search worker.

Bootstrap should install and verify:

- recent NVIDIA driver;
- CUDA >= 12.8 for Blackwell paths;
- Python environment;
- PyTorch;
- Transformers/PEFT/TRL or chosen training stack;
- llama.cpp CUDA build;
- Git LFS if needed;
- benchmark dependencies;
- Docker for agent sandboxes.

The worker must be replaceable. No experiment state should exist only on the rented machine.

## 26. Security model for rented hardware

Use:

- dedicated ephemeral SSH key;
- dedicated unprivileged user;
- repository-scoped GitHub credential;
- narrowly scoped Hugging Face token if required;
- no SSH agent forwarding;
- no broad cloud credentials;
- Docker/network restrictions for model-generated commands;
- credential revocation after worker destruction.

Model-executed sandboxes must not inherit host secrets.

## 27. Experiment storage

Recommended:

- Git for source/config;
- SQLite or DuckDB for metrics;
- object storage or Hugging Face/private artifact storage for checkpoints;
- SHA256 hashes for artifacts;
- optional MLflow/W&B only if useful.

Run directory:

```text
runs/<run-id>/
  candidate.yaml
  env.json
  git_commit.txt
  dataset_manifest.json
  train/
  expert_profile.parquet
  pruning_map.json
  quantization.json
  eval/
  agent_trajectories/
  vram_trace.csv
  speed.json
  artifact_manifest.json
```

Do not commit model weights or run outputs to Git.

## 28. Search stopping criteria

Stop broad search when:

- no Pareto improvement across a configurable number of candidates;
- quality falls before memory target improves;
- all winning candidates cluster around the same pruning/precision regime;
- compute budget is reached.

Then spend remaining compute on:

- repeated evaluation;
- seed robustness;
- recovery tuning;
- release-card validation.

## 29. Likely winning region

Current working expectation:

### Fully resident

- all 40 layers;
- adaptive expert retention, around 50-55% overall;
- Q3 routed experts;
- Q4 for highly sensitive experts;
- Q5/Q6 always-active path;
- Q4 KV;
- model + cache + runtime < 11.5 GiB.

### Quality-first

- about 50-55% expert retention;
- mostly Q4 experts;
- small CPU-MoE offload;
- Q4 KV;
- same 262K release window.

The second may be the better real-world agent even if the first is technically cleaner.

## 30. Milestones

### M0 - foundation

- repo structure;
- config schema;
- worker bootstrap;
- environment lock;
- run database;
- benchmark skeleton;
- VRAM sampler.

### M1 - baselines

- stock reference;
- 12 GB-style 262K runtime baseline;
- low-bit control;
- reproducible metrics.

### M2 - agent/coding tune

- cleaned dataset;
- SFT;
- executable tool-use stage;
- tuned reference checkpoint.

### M3 - profiling and pruning

- expert telemetry;
- pruning search;
- first 50-55% retained models;
- recovery tune.

### M4 - quantization

- mixed-precision search;
- all-GPU candidate;
- quality-first offload candidate.

### M5 - long-context qualification

- 128K;
- 200K;
- 245K-250K prompt tests;
- full 262K allocation;
- repeated stability.

### M6 - release

- actual 12 GB card test;
- final GGUF(s);
- benchmark report;
- exact reproduction manifest.

## 31. Definition of done

The project is done when at least one release candidate:

- runs with `-c 262144`;
- stays below the release VRAM limit;
- can consume roughly 245K-250K tokens and still produce useful output;
- preserves at least 95% of the tuned reference's agent/coding/long-context performance;
- completes iterative tool-use loops reliably;
- survives repeated runs without OOM;
- is reproducible from repository configs and documented artifact hashes.

A second quality-first variant with small CPU expert offload should be released if it performs materially better.
