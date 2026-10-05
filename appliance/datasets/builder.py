from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def dedupe(records: list[dict[str, Any]], key: str = "id") -> list[dict[str, Any]]:
    seen = set()
    result = []
    for record in records:
        value = record.get(key) or json.dumps(record, sort_keys=True)
        if value in seen:
            continue
        seen.add(value)
        result.append(record)
    return result


def weighted_mix(
    groups: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = [name for name in groups if groups[name]]
    if not names:
        return []

    normalized = [max(0.0, weights.get(name, 0.0)) for name in names]
    if sum(normalized) == 0:
        normalized = [1.0] * len(names)

    output = []
    for _ in range(count):
        group = rng.choices(names, weights=normalized, k=1)[0]
        output.append(rng.choice(groups[group]))
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
