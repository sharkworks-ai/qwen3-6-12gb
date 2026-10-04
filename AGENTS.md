# AGENTS.md

## Mission

Build and validate the strongest practical Qwen/Qwen3.6-35B-A3B derivative for autonomous coding and tool use on a single 12 GB GPU with a 262,144-token context window.

Read `docs/PLAN.md` before making architectural changes.

## Hard constraints

- Keep 262,144-token context support.
- Treat 12 GB VRAM as a release constraint, not a development-machine constraint.
- Preserve agentic and coding task success over raw compression ratio.
- Do not use lossy KV eviction in the mainline unless the full-history design cannot meet the target.
- Do not make depth pruning the first compression tool.
- Treat ternary routed experts as experimental until complete tool-loop tests pass.
- Record every experiment configuration and artifact hash.
- Never commit model weights, datasets, credentials, or run outputs.

## Development rules

- Prefer small, composable Python modules.
- All CLI operations must be reproducible from config files.
- Long-running experiments must be resumable.
- Every mutating experiment stage writes a manifest.
- Benchmark scoring must be machine-readable.
- Agent tests must judge repository/test state, not prose alone.
- Keep the release runtime independent from the training framework.

## Security

Follow `docs/SECURITY.md`.

Model-generated commands must run in isolated task containers without host credentials.

## Initial implementation order

1. Hardware/environment probe.
2. Experiment DB and run manifest.
3. VRAM sampler.
4. Runtime baseline runner.
5. Evaluation harness.
6. Dataset manifest/preparation.
7. Training runner.
8. Expert telemetry.
9. Pruning search.
10. Recovery training.
11. Quantization search.
12. Final 12 GB qualification.
