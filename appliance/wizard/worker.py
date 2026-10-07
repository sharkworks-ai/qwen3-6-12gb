"""Execute only registered, fixed pipeline operations in separate processes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from appliance.registry import command_for
from appliance.stages.common import load_config, save_json

ALLOWED = {
    "sft",
    "merge",
    "profile",
    "prune",
    "ayot",
    "full_validation",
    "proof_run",
    "wizard_benchmarks",
}


def run(cfg):
    kind = cfg["job_kind"]
    if kind not in ALLOWED:
        raise ValueError("Operation is not allowed in the automated pipeline")
    stage = Path(cfg["stage_dir"])
    job = dict(cfg["job_config"])
    if kind == "sft":
        job["resume_from_checkpoint"] = bool(
            job.pop("resume", False) and list(Path(job["output_dir"]).glob("checkpoint-*"))
        )
    path = stage / "job-config.json"
    save_json(path, job)
    subprocess.run(command_for(kind, path, job), check=True)
    result = (
        load_config(cfg["result_file"])
        if cfg.get("result_file")
        else {"status": "completed", "kind": kind}
    )
    save_json(stage / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
