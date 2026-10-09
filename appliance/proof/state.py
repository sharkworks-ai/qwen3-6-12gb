"""Durable stage orchestration with per-stage telemetry and output hashes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from appliance.core import NvidiaTelemetry, peak_vram
from appliance.quant.precision import digest
from appliance.runtime.packed import sha256
from appliance.stages.common import load_config, save_json


def hashes(directory):
    return {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.name.endswith(".lock") and not p.name.endswith(".tmp")
    }


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))})


class Stages:
    def __init__(self, cfg, output, identity):
        self.cfg, self.output = cfg, output
        self.path = output / "proof-progress.json"
        self.state = {"identity": identity, "status": "running", "stages": {}}
        if self.path.exists():
            if not cfg.get("resume"):
                raise ValueError("Run exists; select Resume or use a new output directory")
            self.state = load_config(self.path)
            if self.state["identity"] != identity:
                raise ValueError("Proof source/config/version mismatch")
        save_json(self.path, self.state)

    def run(self, name, operation, **options):
        stage = self.output / name
        stage.mkdir(parents=True, exist_ok=True)
        worker = {**self.cfg, **options, "operation": operation, "stage_dir": str(stage)}

        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items() if k != "resume"}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value

        worker_identity = digest(stable(worker))
        old = self.state["stages"].get(name)
        if old and old.get("status") == "succeeded":
            if old["identity"] != worker_identity or hashes(stage) != old["hashes"]:
                raise ValueError(f"Stage integrity failure: {name}")
            return load_config(stage / "result.json")
        config_path = self.output / f"{name}-config.json"
        save_json(config_path, worker)
        self.state["stages"][name] = {"identity": worker_identity, "status": "running"}
        save_json(self.path, self.state)
        start = time.monotonic()
        telemetry = NvidiaTelemetry(stage / "vram.csv")
        telemetry.start()
        code = -1
        interrupted = False
        try:
            for attempt in range(2):
                with (
                    (stage / "stdout.log").open("a") as stdout,
                    (stage / "stderr.log").open("a") as stderr,
                ):
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "appliance.wizard.worker"
                            if operation == "registered"
                            else "appliance.proof.worker",
                            "--config",
                            str(config_path),
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        env=os.environ.copy(),
                    )
                    code = process.returncode
                if code != 75 or attempt == 1:
                    break
                interrupted = True
                worker["resume"] = True
                worker["quant_config"]["resume"] = True
                save_json(config_path, worker)
        finally:
            telemetry.stop()
        selected = str(self.cfg.get("cuda_devices", "0")).split(",")
        peaks = {k: v for k, v in peak_vram(stage / "vram.csv").items() if k in selected}
        item = {
            "identity": worker_identity,
            "status": "succeeded" if code == 0 else "failed",
            "seconds": time.monotonic() - start,
            "peak_vram_mib": peaks,
            "checkpoint_restart_checked": interrupted,
            "exit_code": code,
        }
        if code == 0:
            item["hashes"] = hashes(stage)
        self.state["stages"][name] = item
        if code:
            self.state["status"] = "failed"
        save_json(self.path, self.state)
        print(f"{name}: {item['status']}", flush=True)
        if code:
            raise RuntimeError(
                f"Stage {name} failed ({code}); see {stage}/stderr.log and select Resume"
            )
        return load_config(stage / "result.json")

    def release(self, name, relative):
        """Delete bulky output of a finished stage once every reader has succeeded.

        The stage's integrity hashes are re-baselined so resume still skips it.
        """
        item = self.state["stages"].get(name, {})
        if item.get("status") != "succeeded":
            raise ValueError(f"Cannot release output of unfinished stage: {name}")
        stage = self.output / name
        target = (stage / relative).resolve()
        if stage.resolve() not in target.parents:
            raise ValueError(f"Release path must stay inside stage {name}: {relative}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        released = item.setdefault("released", [])
        if relative not in released:
            released.append(relative)
        item["hashes"] = hashes(stage)
        save_json(self.path, self.state)
