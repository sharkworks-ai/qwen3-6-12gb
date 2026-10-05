from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_trace(root: Path, eval_id: str, trace: list[dict[str, Any]]) -> Path:
    path = root / f"{eval_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path


def summarize_failures(trace: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "invalid_tool_arguments": 0,
        "repeated_failed_commands": 0,
        "fabricated_results": 0,
        "test_failures": 0,
    }
    for event in trace:
        kind = event.get("failure_type")
        if kind in summary:
            summary[kind] += 1
    return summary
