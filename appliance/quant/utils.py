from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command is not installed: {name}")
    return path


def run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=False,
    )


def capture(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_hf_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def model_type(model_dir: Path) -> str | None:
    cfg = load_hf_config(model_dir)
    return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")
