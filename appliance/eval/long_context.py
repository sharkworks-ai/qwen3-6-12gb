from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


LENGTHS = [32768, 65536, 131072, 200000, 250000]


def run_matrix(
    *,
    invoke: Callable[[int], dict],
    output_dir: Path,
    lengths: list[int] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for length in lengths or LENGTHS:
        results[str(length)] = invoke(length)
    (output_dir / "long-context.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
