# OxCoder-9B Benchmark Gate

## Purpose

A Qwen3.6-12GB release is not worthwhile if the extra complexity and compute do not produce a model that clearly surpasses **OrionLLM/OxCoder-9B**.

OxCoder-9B is therefore a hard external baseline, not a reference-only comparison.

Baseline source:

- model: `OrionLLM/OxCoder-9B`
- source: https://huggingface.co/OrionLLM/OxCoder-9B
- snapshot date: 2026-10-05
- OxCoder reports its results as averages over five independent runs.

## Published OxCoder-9B scores

| Domain | Benchmark / harness | OxCoder-9B |
|---|---|---:|
| Coding | Terminal-Bench 2.1 (Terminus-2) | 49.6 |
| Coding | Terminal-Bench 2.1 (Claude Code) | 50.8 |
| Coding | SWE-bench Verified | 73.5 |
| Coding | SWE-bench Pro | 49.1 |
| Coding | NL2Repo | 36.2 |
| Reasoning | HLE, no tools | 21.2 |
| Reasoning | HLE, with tools | 32.8 |
| Reasoning | GPQA Diamond | 86.9 |
| Agentic | MCP-Atlas | 56.7 |
| Agentic | BrowseComp | 57.4 |
| Agentic | ClawEval | 67.8 |

There are 11 published rows in this snapshot: 5 coding, 3 reasoning, and 3 agentic.

## Hard release gate

A release candidate must satisfy **all** of the following:

1. **Overall majority:** score strictly higher than OxCoder-9B on at least **6 of the 11** published rows.
2. **Coding + agentic majority:** score strictly higher on at least **5 of the 8** coding and agentic rows.
3. **Core software-engineering guardrail:** beat OxCoder on at least **one of SWE-bench Verified or SWE-bench Pro**.
4. **Terminal-agent guardrail:** beat OxCoder on at least **one of the two published Terminal-Bench 2.1 harnesses**.
5. Continue to satisfy the project's 12 GB, 262,144-context, stability, tool-schema, and tuned-reference-retention gates.

A tie is not a win.

## Methodology matching

Do not compare scores produced by materially different harnesses and call them wins.

Replicate OxCoder's published settings wherever possible.

### Terminal-Bench 2.1, Terminus-2

Published OxCoder settings:

- Harbor / Terminus-2;
- parser: JSON;
- temperature: 1.0;
- top_p: 1.0;
- 256K context;
- 2-hour task timeout;
- 32 CPU cores;
- 32 GB RAM.

### Terminal-Bench 2.1, Claude Code

Published OxCoder setting:

- Claude Code 2.1.126 harness;
- parser: JSON;
- temperature: 1.0;
- top_p: 1.0.

The exact harness version must be recorded. If the old version cannot be reproduced, report the new result separately rather than silently treating it as directly comparable.

### SWE-bench Verified and SWE-bench Pro

Published OxCoder settings:

- OpenHands harness;
- temperature: 1.0;
- top_p: 0.95;
- 256K context.

Pin dataset versions, harness revisions, container images, and inference parameters for our comparison runs.

## Statistical policy

OxCoder reports five-run averages. For any benchmark with meaningful stochasticity, our finalist comparison should also use repeated runs.

Minimum policy:

- five runs where practical and where OxCoder reports a five-run mean;
- fixed seeds recorded;
- report mean and standard deviation;
- a nominal win smaller than ordinary run-to-run variance should be flagged as marginal.

For deterministic or prohibitively expensive suites, record the official harness behavior and exact run count.

## Benchmark leakage policy

These 11 benchmark evaluation sets are **release tests**.

Do not:

- train directly on their held-out test instances;
- generate recovery data from their test solutions;
- use finalist scores as per-example optimization rewards.

It is acceptable to use training data from the same broad task domains, but evaluation contamination must be actively checked.

## Search integration

The experiment controller must record:

```yaml
oxcoder:
  overall_wins: integer
  coding_agentic_wins: integer
  swe_guardrail_pass: boolean
  terminal_guardrail_pass: boolean
  benchmark_results:
    terminal_bench_2_1_terminus2: number|null
    terminal_bench_2_1_claude_code: number|null
    swe_bench_verified: number|null
    swe_bench_pro: number|null
    nl2repo: number|null
    hle_no_tools: number|null
    hle_with_tools: number|null
    gpqa_diamond: number|null
    mcp_atlas: number|null
    browsecomp: number|null
    claweval: number|null
```

Only finalist-tier candidates need the complete expensive 11-row suite.

Earlier tiers should use cheaper proxy tests that correlate with these targets.

## Definition of success

The project has produced a worthwhile model only when a 12 GB / 262K release candidate passes this OxCoder dominance gate.

If no candidate can do so without violating the hardware or context requirements, the correct project result is **no release**, rather than publishing a weaker model simply because it fits.
