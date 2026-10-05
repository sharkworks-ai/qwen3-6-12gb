from __future__ import annotations

import csv
import hmac
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from itsdangerous import BadSignature, URLSafeSerializer

from appliance.registry import JOBS, command_for, get_job


@dataclass(frozen=True)
class Settings:
    data_root: Path
    web_token: str
    github_token: str | None
    hf_token: str | None
    secure_cookie: bool

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("QWEN12G_WEB_TOKEN", "")
        if len(token) < 12:
            raise RuntimeError("QWEN12G_WEB_TOKEN must be at least 12 characters")
        return cls(
            data_root=Path(os.environ.get("QWEN12G_DATA_ROOT", "/data")),
            web_token=token,
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            hf_token=os.environ.get("HF_TOKEN") or None,
            secure_cookie=os.environ.get("QWEN12G_SECURE_COOKIE", "0") == "1",
        )

    def ensure_dirs(self) -> None:
        for name in ("db", "runs", "models", "datasets", "checkpoints", "artifacts", "hf-cache", "repos"):
            (self.data_root / name).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Job:
    run_id: str
    kind: str
    status: str
    pid: int | None
    created_at: str
    updated_at: str
    exit_code: int | None
    config: dict[str, Any]


class ApplianceDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                pid INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                exit_code INTEGER, config_json TEXT NOT NULL)""")

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def create_job(self, run_id: str, kind: str, config: dict[str, Any]) -> Job:
        now = datetime.now(UTC).isoformat()
        with self.connect() as con:
            con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                        (run_id, kind, "queued", None, now, now, None, json.dumps(config)))
        return self.get_job(run_id)

    def update_job(self, run_id: str, *, status: str | None = None,
                   pid: int | None = None, exit_code: int | None = None) -> None:
        job = self.get_job(run_id)
        with self.connect() as con:
            con.execute("UPDATE jobs SET status=?,pid=?,updated_at=?,exit_code=? WHERE run_id=?",
                        (status if status is not None else job.status,
                         pid if pid is not None else job.pid,
                         datetime.now(UTC).isoformat(),
                         exit_code if exit_code is not None else job.exit_code,
                         run_id))

    def get_job(self, run_id: str) -> Job:
        with self.connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return Job(row["run_id"], row["kind"], row["status"], row["pid"],
                   row["created_at"], row["updated_at"], row["exit_code"], json.loads(row["config_json"]))

    def list_jobs(self, limit: int = 100) -> list[Job]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job(r["run_id"], r["kind"], r["status"], r["pid"], r["created_at"],
                    r["updated_at"], r["exit_code"], json.loads(r["config_json"])) for r in rows]


class Auth:
    COOKIE = "qwen12g_session"
    def __init__(self, secret: str):
        self.secret = secret
        self.serializer = URLSafeSerializer(secret, salt="qwen12g-web")
    def verify_token(self, token: str) -> bool:
        return hmac.compare_digest(token, self.secret)
    def cookie_value(self) -> str:
        return self.serializer.dumps({"authenticated": True})
    def verify_cookie(self, value: str | None) -> bool:
        if not value: return False
        try: payload = self.serializer.loads(value)
        except BadSignature: return False
        return payload.get("authenticated") is True


def gpu_info() -> list[dict[str, str]]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError): return []
    rows=[]
    for line in result.stdout.splitlines():
        p=[x.strip() for x in line.split(",")]
        if len(p)>=6: rows.append(dict(index=p[0],name=p[1],memory_total=p[2],memory_used=p[3],utilization=p[4],temperature=p[5]))
    return rows


def storage_info(path: Path) -> dict[str, int]:
    u=shutil.disk_usage(path); return dict(total=u.total, used=u.used, free=u.free)


class NvidiaTelemetry:
    QUERY="index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    def __init__(self, output: Path, interval: float=1.0):
        self.output=output; self.interval=interval; self.stop_event=threading.Event(); self.thread=None
    def start(self):
        self.output.parent.mkdir(parents=True, exist_ok=True); self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread and self.thread.join(timeout=5)
    def _run(self):
        with self.output.open("w",newline="",encoding="utf-8") as h:
            w=csv.writer(h); w.writerow(["timestamp","index","name","memory_total_mib","memory_used_mib","utilization_gpu_percent","temperature_c","power_w"])
            while not self.stop_event.is_set():
                now=datetime.now(UTC).isoformat()
                try:
                    r=subprocess.run(["nvidia-smi",f"--query-gpu={self.QUERY}","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
                    for line in r.stdout.splitlines():
                        if line.strip(): w.writerow([now,*[x.strip() for x in line.split(",")]])
                    h.flush()
                except (OSError,subprocess.CalledProcessError) as e: w.writerow([now,"error",str(e)]); h.flush()
                self.stop_event.wait(self.interval)


def peak_vram(path: Path) -> dict[str,float]:
    peaks={}
    if not path.exists(): return peaks
    with path.open("r",encoding="utf-8",newline="") as h:
        for row in csv.DictReader(h):
            idx,used=row.get("index"),row.get("memory_used_mib")
            if not idx or idx=="error" or not used: continue
            try: value=float(used)
            except ValueError: continue
            peaks[idx]=max(peaks.get(idx,0.0),value)
    return peaks


class JobManager:
    def __init__(self, db: ApplianceDB, runs_root: Path): self.db=db; self.runs_root=runs_root
    def start(self, kind: str, config: dict | None=None) -> str:
        get_job(kind); config=config or {}
        run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")+f"-{kind}-"+uuid4().hex[:8]
        run_dir=self.runs_root/run_id; run_dir.mkdir(parents=True,exist_ok=False)
        config_path=run_dir/"config.json"; config_path.write_text(json.dumps(config,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        self.db.create_job(run_id,kind,config)
        process=subprocess.Popen([sys.executable,"-m","appliance.runner","--run-id",run_id,"--kind",kind,"--run-root",str(self.runs_root)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,text=True)
        self.db.update_job(run_id,status="running",pid=process.pid); return run_id
    def reconcile(self):
        for job in self.db.list_jobs(1000):
            if job.status not in {"running","stopping"}: continue
            result_path=self.runs_root/job.run_id/"result.json"
            if result_path.exists():
                result=json.loads(result_path.read_text(encoding="utf-8")); self.db.update_job(job.run_id,status=result["status"],exit_code=result["exit_code"]); continue
            if job.pid is None: self.db.update_job(job.run_id,status="interrupted"); continue
            try: os.kill(job.pid,0)
            except ProcessLookupError: self.db.update_job(job.run_id,status="interrupted")
    def stop(self, run_id: str):
        job=self.db.get_job(run_id)
        if job.pid is None: return
        try: os.killpg(job.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        self.db.update_job(run_id,status="stopping")
