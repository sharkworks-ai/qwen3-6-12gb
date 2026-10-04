from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen12g.jobs import remote_storage
from qwen12g.worker import WorkerConfig, gpu_inventory, remote_info


def _command_version(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return (result.stdout or result.stderr).strip()


def local_environment() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "git_version": _command_version(["git", "--version"]),
        "docker_version": _command_version(["docker", "--version"]),
    }


def remote_environment(config: WorkerConfig) -> dict[str, Any]:
    info_result = remote_info(config)
    info = json.loads(info_result.stdout)
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "worker": config.name,
        "docker_context": config.docker_context,
        "docker": {
            "name": info.get("Name"),
            "server_version": info.get("ServerVersion"),
            "operating_system": info.get("OperatingSystem"),
            "architecture": info.get("Architecture"),
            "cpus": info.get("NCPU"),
            "memory_bytes": info.get("MemTotal"),
        },
        "gpus": gpu_inventory(config),
        "storage": remote_storage(config),
    }


def full_environment(config: WorkerConfig) -> dict[str, Any]:
    return {
        "local": local_environment(),
        "remote": remote_environment(config),
    }


def write_probe(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
