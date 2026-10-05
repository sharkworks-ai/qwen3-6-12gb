from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .base import EvalAdapter, EvalContext


class CommandEval(EvalAdapter):
    """Pinned external benchmark adapter.

    Commands are config-defined but benchmark names are code-registered, avoiding
    a general UI shell.
    """

    def __init__(self, name: str, executable: str) -> None:
        self.name = name
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def run(self, context: EvalContext) -> dict:
        command = context.config.get("commands", {}).get(self.name)
        if not command:
            raise RuntimeError(f"No pinned command configured for {self.name}")
        context.output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        (context.output_dir / f"{self.name}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (context.output_dir / f"{self.name}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        payload = {
            "name": self.name,
            "exit_code": completed.returncode,
            "status": "succeeded" if completed.returncode == 0 else "failed",
        }
        (context.output_dir / f"{self.name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload


REGISTERED = {
    "swe_bench_verified": CommandEval("swe_bench_verified", "python3"),
    "swe_bench_pro": CommandEval("swe_bench_pro", "python3"),
    "terminal_bench": CommandEval("terminal_bench", "python3"),
    "nl2repo": CommandEval("nl2repo", "python3"),
    "mcp_atlas": CommandEval("mcp_atlas", "python3"),
    "browsecomp": CommandEval("browsecomp", "python3"),
    "claweval": CommandEval("claweval", "python3"),
    "gpqa_diamond": CommandEval("gpqa_diamond", "python3"),
    "hle": CommandEval("hle", "python3"),
}
