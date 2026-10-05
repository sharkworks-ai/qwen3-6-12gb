from __future__ import annotations

import itertools
from hashlib import sha256
from typing import Any


def stable_candidate_id(prefix: str, config: dict[str, Any]) -> str:
    digest = sha256(repr(sorted(config.items())).encode()).hexdigest()[:10]
    return f"{prefix}-{digest}"


def generate_quant_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    backends = base.get("backends", ["llama_cpp", "turboquant"])
    expert_quants = base.get("expert_quants", ["q4", "q3", "mixed_q3_q4"])
    kv_options = base.get(
        "kv_options",
        [
            {"k": "q4_0", "v": "q4_0"},
            {"k": "q8_0", "v": "turbo3"},
        ],
    )
    results = []
    for backend, expert_quant, kv in itertools.product(
        backends, expert_quants, kv_options
    ):
        config = dict(
            backend=backend,
            expert_quant=expert_quant,
            cache_k=kv["k"],
            cache_v=kv["v"],
        )
        results.append(
            {
                "candidate_id": stable_candidate_id("quant", config),
                "stage": "quantize",
                "config": config,
            }
        )
    return results


def generate_prune_candidates(base: dict[str, Any]) -> list[dict[str, Any]]:
    retention = base.get("retention", [0.75, 0.625, 0.5])
    profiles = base.get("profiles", ["uniform", "adaptive"])
    results = []
    for keep, profile in itertools.product(retention, profiles):
        config = {"retention": keep, "profile": profile}
        results.append(
            {
                "candidate_id": stable_candidate_id("prune", config),
                "stage": "prune",
                "config": config,
            }
        )
    return results
