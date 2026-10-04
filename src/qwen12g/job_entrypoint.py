from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from qwen12g.manifest import write_json
from qwen12g.telemetry import NvidiaSmiSampler, peak_memory_by_gpu

STAGES: dict[str, list[str]] = {
    "smoke": [sys.executable, "-m", "qwen12g.smoke_job"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--run-root", default="/qwen-data/runs")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--image-id", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC).isoformat()
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "stage": args.stage,
            "git_commit": args.git_commit,
            "image_id": args.image_id,
            "status": "running",
            "started_at": started,
        },
    )

    telemetry_path = run_dir / "vram_trace.csv"
    sampler = NvidiaSmiSampler(telemetry_path)
    sampler.start()

    command = STAGES[args.stage]
    exit_code = 1
    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
            exit_code = completed.returncode
    finally:
        sampler.stop()

    finished = datetime.now(UTC).isoformat()
    result = {
        "run_id": args.run_id,
        "stage": args.stage,
        "git_commit": args.git_commit,
        "image_id": args.image_id,
        "status": "succeeded" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_at": started,
        "finished_at": finished,
        "peak_vram_mib": peak_memory_by_gpu(telemetry_path),
    }
    write_json(run_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
