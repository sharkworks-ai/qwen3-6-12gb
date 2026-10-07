from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from appliance.core import NvidiaTelemetry, gpu_info, peak_vram
from appliance.registry import command_for, get_job


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); parser.add_argument("--kind",required=True); parser.add_argument("--run-root",default="/data/runs"); args=parser.parse_args()
    get_job(args.kind)
    run_dir=Path(args.run_root)/args.run_id; config_path=run_dir/"config.json"
    config=json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    started=datetime.now(UTC).isoformat()
    (run_dir/"run.json").write_text(json.dumps({"run_id":args.run_id,"kind":args.kind,"status":"running","started_at":started,"config":config,"gpu_backend":os.environ.get("QWEN12G_GPU_BACKEND", "auto"),"gpu_inventory":gpu_info()},indent=2)+"\n",encoding="utf-8")
    env=os.environ.copy()
    if config.get("cuda_devices") is not None: env["CUDA_VISIBLE_DEVICES"]=str(config["cuda_devices"])
    telemetry=NvidiaTelemetry(run_dir/"vram_trace.csv"); telemetry.start(); returncode=1
    try:
        with (run_dir/"stdout.log").open("w",encoding="utf-8") as stdout,(run_dir/"stderr.log").open("w",encoding="utf-8") as stderr:
            command=command_for(args.kind,config_path,config)
            p=subprocess.run(command,stdout=stdout,stderr=stderr,text=True,check=False,env=env); returncode=p.returncode
    finally: telemetry.stop()
    result={"run_id":args.run_id,"kind":args.kind,"status":"succeeded" if returncode==0 else "failed","exit_code":returncode,"started_at":started,"finished_at":datetime.now(UTC).isoformat(),"peak_vram_mib":peak_vram(run_dir/"vram_trace.csv")}
    (run_dir/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); raise SystemExit(returncode)

if __name__=="__main__": main()
