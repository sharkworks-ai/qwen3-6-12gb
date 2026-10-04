from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen12g.db import RunDB, RunRecord
from qwen12g.manifest import git_commit, new_run_id
from qwen12g.worker import WorkerConfig, docker_args, run_command


PROJECT_LABEL = "qwen3.6-12gb"
DEFAULT_IMAGE = "qwen12g-worker:local"
ALLOWED_STAGES = {"smoke"}
ALLOWED_GPU_MODES = {"distributed", "gpu0", "gpu1"}


@dataclass(frozen=True)
class RemoteStatus:
    status: str
    exit_code: int | None
    started_at: str | None
    finished_at: str | None


def _require_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is dirty. Commit or stash changes before launching a remote run."
        )


def _image_id(config: WorkerConfig, image: str) -> str:
    result = run_command(
        docker_args(config, "image", "inspect", image, "--format", "{{.Id}}"),
        capture_output=True,
    )
    return result.stdout.strip()


def _gpu_argument(mode: str) -> str:
    if mode == "distributed":
        return "all"
    if mode == "gpu0":
        return "device=0"
    if mode == "gpu1":
        return "device=1"
    raise ValueError(f"Unsupported GPU mode: {mode}")


def _safe_container_name(run_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_id)
    return f"qwen12g-{cleaned}"[:120]


def launch_stage(
    config: WorkerConfig,
    *,
    stage: str,
    gpu_mode: str = "distributed",
    db: RunDB | None = None,
    image: str = DEFAULT_IMAGE,
) -> RunRecord:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Stage is not registered for remote execution: {stage}")
    if gpu_mode not in ALLOWED_GPU_MODES:
        raise ValueError(f"Unsupported GPU mode: {gpu_mode}")

    _require_clean_git()
    commit = git_commit()
    image_id = _image_id(config, image)
    run_id = new_run_id(stage)
    container_name = _safe_container_name(run_id)
    created_at = datetime.now(UTC).isoformat()
    remote_run_dir = f"{config.data_root}/runs/{run_id}"

    args = docker_args(
        config,
        "run",
        "-d",
        "--name",
        container_name,
        "--label",
        f"qwen12g.project={PROJECT_LABEL}",
        "--label",
        f"qwen12g.run_id={run_id}",
        "--label",
        f"qwen12g.stage={stage}",
        "--label",
        f"qwen12g.git_commit={commit}",
        "--gpus",
        _gpu_argument(gpu_mode),
        "--ipc",
        "host",
        "--shm-size",
        "32g",
        "-e",
        "HF_HOME=/qwen-data/hf-cache",
        "-e",
        "QWEN12G_RUN_ROOT=/qwen-data/runs",
        "-v",
        f"{config.data_root}/hf-cache:/qwen-data/hf-cache",
        "-v",
        f"{config.data_root}/datasets:/qwen-data/datasets",
        "-v",
        f"{config.data_root}/checkpoints:/qwen-data/checkpoints",
        "-v",
        f"{config.data_root}/artifacts:/qwen-data/artifacts",
        "-v",
        f"{config.data_root}/runs:/qwen-data/runs",
        image,
        "python3",
        "-m",
        "qwen12g.job_entrypoint",
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--git-commit",
        commit,
        "--image-id",
        image_id,
    )

    result = run_command(args, capture_output=True)
    container_id = result.stdout.strip()

    record = RunRecord(
        run_id=run_id,
        stage=stage,
        status="running",
        worker=config.name,
        docker_context=config.docker_context,
        container_name=container_name,
        container_id=container_id,
        git_commit=commit,
        image=image,
        image_id=image_id,
        gpu_mode=gpu_mode,
        created_at=created_at,
        updated_at=created_at,
        remote_run_dir=remote_run_dir,
        metadata={"launch_args_kind": "registered-stage"},
    )
    (db or RunDB()).insert(record)
    return record


def inspect_run(config: WorkerConfig, record: RunRecord) -> RemoteStatus:
    result = run_command(
        docker_args(
            config,
            "inspect",
            record.container_name,
            "--format",
            "{{json .State}}",
        ),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return RemoteStatus("missing", None, None, None)

    state = json.loads(result.stdout)
    docker_status = state.get("Status", "unknown")
    exit_code = state.get("ExitCode")
    if docker_status == "exited":
        status = "succeeded" if exit_code == 0 else "failed"
    elif docker_status in {"running", "created", "restarting", "paused"}:
        status = docker_status
    else:
        status = docker_status

    return RemoteStatus(
        status=status,
        exit_code=exit_code if docker_status == "exited" else None,
        started_at=state.get("StartedAt"),
        finished_at=state.get("FinishedAt"),
    )


def sync_run_status(
    config: WorkerConfig,
    run_id: str,
    *,
    db: RunDB | None = None,
) -> RunRecord:
    database = db or RunDB()
    record = database.get(run_id)
    if record is None:
        raise KeyError(run_id)

    remote = inspect_run(config, record)
    database.update_status(
        run_id,
        remote.status,
        metadata_update={
            "exit_code": remote.exit_code,
            "remote_started_at": remote.started_at,
            "remote_finished_at": remote.finished_at,
        },
    )
    updated = database.get(run_id)
    if updated is None:
        raise KeyError(run_id)
    return updated


def stop_run(config: WorkerConfig, record: RunRecord, timeout: int = 30) -> None:
    run_command(
        docker_args(
            config,
            "stop",
            "--time",
            str(timeout),
            record.container_name,
        ),
        check=False,
    )


def run_logs(
    config: WorkerConfig,
    record: RunRecord,
    *,
    tail: int = 200,
) -> str:
    result = run_command(
        docker_args(
            config,
            "logs",
            "--tail",
            str(tail),
            record.container_name,
        ),
        check=False,
        capture_output=True,
    )
    return result.stdout + result.stderr


def read_remote_run_file(
    config: WorkerConfig,
    run_id: str,
    relative_path: str,
    *,
    image: str = DEFAULT_IMAGE,
) -> str | None:
    if ".." in Path(relative_path).parts or Path(relative_path).is_absolute():
        raise ValueError("relative_path must stay inside the run directory")

    result = run_command(
        docker_args(
            config,
            "run",
            "--rm",
            "-v",
            f"{config.data_root}/runs:/qwen-data/runs:ro",
            image,
            "cat",
            f"/qwen-data/runs/{run_id}/{relative_path}",
        ),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def remote_storage(config: WorkerConfig, image: str = DEFAULT_IMAGE) -> dict[str, int]:
    result = run_command(
        docker_args(
            config,
            "run",
            "--rm",
            "-v",
            f"{config.data_root}:/qwen-data",
            image,
            "df",
            "-B1",
            "--output=size,used,avail",
            "/qwen-data",
        ),
        capture_output=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected df output: {result.stdout!r}")
    size, used, available = (int(value) for value in lines[-1].split())
    return {"size_bytes": size, "used_bytes": used, "available_bytes": available}


def parse_result(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    return json.loads(text)
