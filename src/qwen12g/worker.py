from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import yaml


@dataclass(frozen=True)
class WorkerConfig:
    name: str
    docker_context: str
    compose_file: Path
    service: str
    data_root: str
    expected_gpus: int
    expected_gpu_model: str | None = None


def load_worker_config(path: Path) -> WorkerConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    worker = raw["worker"]
    return WorkerConfig(
        name=worker["name"],
        docker_context=worker["docker_context"],
        compose_file=Path(worker["compose_file"]),
        service=worker.get("service", "trainer"),
        data_root=worker.get("data_root", "/srv/qwen12g"),
        expected_gpus=int(worker.get("expected_gpus", 2)),
        expected_gpu_model=worker.get("expected_gpu_model"),
    )


def run_command(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def docker_args(config: WorkerConfig, *args: str) -> list[str]:
    return ["docker", "--context", config.docker_context, *args]


def compose_args(config: WorkerConfig, *args: str) -> list[str]:
    return docker_args(
        config,
        "compose",
        "-f",
        str(config.compose_file),
        *args,
    )


def create_context(context_name: str, ssh_host: str) -> None:
    run_command(
        [
            "docker",
            "context",
            "create",
            context_name,
            "--docker",
            f"host=ssh://{ssh_host}",
        ]
    )


def inspect_context(config: WorkerConfig) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["docker", "context", "inspect", config.docker_context],
        capture_output=True,
    )


def remote_info(config: WorkerConfig) -> subprocess.CompletedProcess[str]:
    return run_command(
        docker_args(
            config,
            "info",
            "--format",
            "{{json .}}",
        ),
        capture_output=True,
    )


def compose_config(config: WorkerConfig) -> subprocess.CompletedProcess[str]:
    return run_command(
        compose_args(config, "config"),
        capture_output=True,
    )


def build_worker(config: WorkerConfig) -> None:
    run_command(compose_args(config, "build", config.service))


def start_worker(config: WorkerConfig) -> None:
    run_command(compose_args(config, "up", "-d", config.service))


def stop_worker(config: WorkerConfig) -> None:
    run_command(compose_args(config, "down", "--remove-orphans"))


def worker_logs(config: WorkerConfig, tail: int = 200) -> None:
    run_command(
        compose_args(
            config,
            "logs",
            "--tail",
            str(tail),
            config.service,
        )
    )


def gpu_inventory(config: WorkerConfig) -> list[str]:
    result = run_command(
        compose_args(
            config,
            "run",
            "--rm",
            config.service,
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
