from __future__ import annotations

from typing import Any


def assign_precision(
    sensitivities: dict[str, float],
    *,
    q4_threshold: float = 0.75,
    q3_threshold: float = 0.30,
) -> dict[str, str]:
    """Turn normalized tensor sensitivity into a mixed-precision plan."""
    result = {}
    for tensor, score in sensitivities.items():
        if score >= q4_threshold:
            result[tensor] = "Q5_K"
        elif score >= q3_threshold:
            result[tensor] = "Q4_K"
        else:
            result[tensor] = "Q3_K"
    return result
