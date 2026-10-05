from __future__ import annotations

from typing import Any


MAXIMIZE = (
    "agent_score",
    "coding_score",
    "long_context_score",
    "decode_tps",
    "oxcoder_overall_wins",
)
MINIMIZE = (
    "peak_vram_mib",
    "cpu_offload_gib",
    "prefill_seconds",
    "artifact_size_bytes",
)


def _value(metrics: dict[str, Any], key: str, maximize: bool) -> float:
    value = metrics.get(key)
    if value is None:
        return float("-inf") if maximize else float("inf")
    return float(value)


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when a is no worse on every available objective and better on one."""
    at_least = True
    strict = False

    for key in MAXIMIZE:
        av = _value(a, key, True)
        bv = _value(b, key, True)
        if av < bv:
            at_least = False
        if av > bv:
            strict = True

    for key in MINIMIZE:
        av = _value(a, key, False)
        bv = _value(b, key, False)
        if av > bv:
            at_least = False
        if av < bv:
            strict = True

    return at_least and strict


def frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        if any(
            dominates(other.get("metrics", {}), metrics)
            for other in candidates
            if other["candidate_id"] != candidate["candidate_id"]
        ):
            continue
        result.append(candidate)
    return result
