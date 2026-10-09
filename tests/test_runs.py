import subprocess
from pathlib import Path

import pytest

from qwen12g import jobs
from qwen12g.db import RunDB, RunRecord
from qwen12g.jobs import _gpu_argument, parse_result
from qwen12g.manifest import new_run_id, write_json
from qwen12g.worker import load_worker_config


def test_new_run_id_contains_stage() -> None:
    run_id = new_run_id("smoke")
    assert "-smoke-" in run_id


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("distributed", "all"),
        ("gpu0", "device=0"),
        ("gpu1", "device=1"),
    ],
)
def test_gpu_argument(mode: str, expected: str) -> None:
    assert _gpu_argument(mode) == expected


def test_run_db_round_trip(tmp_path: Path) -> None:
    db = RunDB(tmp_path / "runs.sqlite3")
    record = RunRecord(
        run_id="run-1",
        stage="smoke",
        status="running",
        worker="dual5090",
        docker_context="qwen5090",
        container_name="qwen12g-run-1",
        container_id="abc",
        git_commit="deadbeef",
        image="worker",
        image_id="sha256:1",
        gpu_mode="distributed",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        remote_run_dir="/srv/qwen12g/runs/run-1",
        metadata={"x": 1},
    )
    db.insert(record)
    loaded = db.get("run-1")
    assert loaded == record

    db.update_status("run-1", "succeeded", metadata_update={"exit_code": 0})
    loaded = db.get("run-1")
    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.metadata["exit_code"] == 0


def test_write_json_is_parseable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, {"ok": True})
    assert parse_result(path.read_text(encoding="utf-8")) == {"ok": True}


def test_read_remote_run_file_bypasses_image_entrypoint(monkeypatch) -> None:
    config = load_worker_config(Path("configs/worker/dual5090.yaml"))
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, '{"status": "succeeded"}', "")

    monkeypatch.setattr(jobs, "run_command", fake_run)
    text = jobs.read_remote_run_file(config, "run-1", "result.json")
    assert parse_result(text) == {"status": "succeeded"}
    args = seen["args"]
    assert args[args.index("--entrypoint") + 1] == "cat"
    assert args.index("--entrypoint") < args.index(jobs.DEFAULT_IMAGE)
