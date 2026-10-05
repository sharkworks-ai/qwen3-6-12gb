from __future__ import annotations

import itertools
from typing import Any


def runtime_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    runtimes = config.get(
        "runtimes",
        ["llama_cpp", "turboquant"],
    )
    kv = config.get(
        "kv",
        [
            {"k": "q4_0", "v": "q4_0"},
            {"k": "q8_0", "v": "q8_0"},
            {"k": "q8_0", "v": "turbo3"},
        ],
    )
    flash = config.get("flash_attention", [True, False])
    cpu_moe = config.get("cpu_moe_layers", [0, 4, 8])

    output = []
    for runtime, kv_cfg, fa, offload in itertools.product(
        runtimes, kv, flash, cpu_moe
    ):
        output.append(
            {
                "runtime": runtime,
                "cache_k": kv_cfg["k"],
                "cache_v": kv_cfg["v"],
                "flash_attention": fa,
                "cpu_moe_layers": offload,
            }
        )
    return output
