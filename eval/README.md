# Evaluation Harness

The evaluation system is the project's quality gate.

## Suites

### coding

Fast screening plus repository-scale tasks.

### agents

Multi-turn tool-use tasks executed in isolated containers. Score actual task completion, tests, and repository state.

### long_context

Run at 32K, 64K, 128K, ~200K, and 245K-250K prompt lengths inside the 262,144-token window.

### general

Small retention suite for reasoning and practical knowledge used by coding agents.

## Tiers

1. smoke
2. cheap
3. normal
4. expensive
5. finalist

Do not run the most expensive 262K tests against every candidate.

## Required finalist checks

- allocate 262,144 context;
- process ~245K-250K prompt with output headroom;
- complete full tool loop;
- run five repeated sessions without OOM;
- record peak VRAM and throughput;
- compare against tuned uncompressed reference.
