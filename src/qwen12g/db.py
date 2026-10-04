from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    override = os.environ.get("QWEN12G_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / "qwen12g" / "runs.sqlite3"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    stage: str
    status: str
    worker: str
    docker_context: str
    container_name: str
    container_id: str | None
    git_commit: str
    image: str
    image_id: str | None
    gpu_mode: str
    created_at: str
    updated_at: str
    remote_run_dir: str
    metadata: dict[str, Any]


class RunDB:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker TEXT NOT NULL,
                    docker_context TEXT NOT NULL,
                    container_name TEXT NOT NULL,
                    container_id TEXT,
                    git_commit TEXT NOT NULL,
                    image TEXT NOT NULL,
                    image_id TEXT,
                    gpu_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    remote_run_dir TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)"
            )

    def insert(self, record: RunRecord) -> None:
        data = asdict(record)
        metadata = json.dumps(data.pop("metadata"), sort_keys=True)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, stage, status, worker, docker_context, container_name,
                    container_id, git_commit, image, image_id, gpu_mode,
                    created_at, updated_at, remote_run_dir, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*data.values(), metadata),
            )

    def get(self, run_id: str) -> RunRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, limit: int = 50) -> list[RunRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        container_id: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> None:
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)

        metadata = dict(record.metadata)
        if metadata_update:
            metadata.update(metadata_update)

        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, updated_at = ?, container_id = COALESCE(?, container_id),
                    metadata_json = ?
                WHERE run_id = ?
                """,
                (status, now, container_id, json.dumps(metadata, sort_keys=True), run_id),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            stage=row["stage"],
            status=row["status"],
            worker=row["worker"],
            docker_context=row["docker_context"],
            container_name=row["container_name"],
            container_id=row["container_id"],
            git_commit=row["git_commit"],
            image=row["image"],
            image_id=row["image_id"],
            gpu_mode=row["gpu_mode"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            remote_run_dir=row["remote_run_dir"],
            metadata=json.loads(row["metadata_json"]),
        )
