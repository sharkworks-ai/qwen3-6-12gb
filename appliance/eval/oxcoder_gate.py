from __future__ import annotations

BASELINE = {
    "terminal_bench_terminus2": 49.6,
    "terminal_bench_claude_code": 50.8,
    "swe_bench_verified": 73.5,
    "swe_bench_pro": 49.1,
    "nl2repo": 36.2,
    "hle_no_tools": 21.2,
    "hle_with_tools": 32.8,
    "gpqa_diamond": 86.9,
    "mcp_atlas": 56.7,
    "browsecomp": 57.4,
    "claweval": 67.8,
}

CODING_AGENTIC = {
    "terminal_bench_terminus2",
    "terminal_bench_claude_code",
    "swe_bench_verified",
    "swe_bench_pro",
    "nl2repo",
    "mcp_atlas",
    "browsecomp",
    "claweval",
}


def score(results: dict[str, float]) -> dict:
    wins = {
        name: results[name] > baseline
        for name, baseline in BASELINE.items()
        if name in results
    }
    overall = sum(wins.values())
    coding_agentic = sum(v for k, v in wins.items() if k in CODING_AGENTIC)
    swe = any(
        wins.get(name, False)
        for name in ("swe_bench_verified", "swe_bench_pro")
    )
    terminal = any(
        wins.get(name, False)
        for name in (
            "terminal_bench_terminus2",
            "terminal_bench_claude_code",
        )
    )
    return {
        "wins": wins,
        "overall_wins": overall,
        "coding_agentic_wins": coding_agentic,
        "overall_gate": overall >= 6,
        "coding_agentic_gate": coding_agentic >= 5,
        "swe_guardrail": swe,
        "terminal_guardrail": terminal,
        "pass": overall >= 6 and coding_agentic >= 5 and swe and terminal,
    }
