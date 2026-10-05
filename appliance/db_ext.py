from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkbenchDB:
    """Extra workbench tables kept separate from the legacy job table."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_path TEXT,
                    runtime TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lineage (
                    child_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (child_id, parent_id, relation)
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    license_json TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eval_runs (
                    eval_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    suite TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    trace_path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS secrets (
                    name TEXT PRIMARY KEY,
                    ciphertext BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    event TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                """
            )

    def upsert_candidate(self, payload: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO candidates (
                    candidate_id,parent_id,stage,status,config_json,metrics_json,
                    artifact_path,runtime,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    parent_id=excluded.parent_id,
                    stage=excluded.stage,
                    status=excluded.status,
                    config_json=excluded.config_json,
                    metrics_json=excluded.metrics_json,
                    artifact_path=excluded.artifact_path,
                    runtime=excluded.runtime,
                    updated_at=excluded.updated_at
                """,
                (
                    payload["candidate_id"],
                    payload.get("parent_id"),
                    payload["stage"],
                    payload.get("status", "created"),
                    json.dumps(payload.get("config", {}), sort_keys=True),
                    json.dumps(payload.get("metrics", {}), sort_keys=True),
                    payload.get("artifact_path"),
                    payload.get("runtime"),
                    now,
                    now,
                ),
            )

    def list_candidates(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM candidates ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_candidate(r) for r in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return self._decode_candidate(row)

    @staticmethod
    def _decode_candidate(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "candidate_id": row["candidate_id"],
            "parent_id": row["parent_id"],
            "stage": row["stage"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "artifact_path": row["artifact_path"],
            "runtime": row["runtime"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def add_lineage(self, child_id: str, parent_id: str, relation: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO lineage VALUES (?,?,?,?)",
                (child_id, parent_id, relation, datetime.now(UTC).isoformat()),
            )

    def lineage_for(self, candidate_id: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM lineage
                WHERE child_id=? OR parent_id=?
                ORDER BY created_at
                """,
                (candidate_id, candidate_id),
            ).fetchall()
        return [dict(r) for r in rows]
