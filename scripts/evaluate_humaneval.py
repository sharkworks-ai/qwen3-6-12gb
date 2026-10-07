"""Reproducible sampled coding diagnostic with isolated execution of generated code."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def execute(code, task, image):
    with tempfile.TemporaryDirectory(prefix="qwen-eval-") as directory:
        path = Path(directory) / "task.py"
        path.write_text(code + "\n" + task["test"] + "\ncheck(" + task["entry_point"] + ")\n")
        path.chmod(0o644)
        name = "qwen-eval-" + uuid.uuid4().hex[:12]
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=256m",
            "--memory-swap=256m",
            "--cpus=1",
            "--pids-limit=64",
            "--user=65534:65534",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=32m",
            "--mount",
            f"type=bind,src={path},dst=/task.py,readonly",
            image,
            "python",
            "-I",
            "/task.py",
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=15, check=False
            )
            return {
                "passed": process.returncode == 0,
                "exit_code": process.returncode,
                "stderr": process.stderr[-2000:],
            }
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return {"passed": False, "error": "sandbox_timeout"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg["dataset"])
    with gzip.open(dataset, "rt") as stream:
        problems = [json.loads(line) for line in stream]
    selected = random.Random(cfg.get("seed", 42)).sample(problems, cfg.get("samples", 20))
    results_path = output / "results.jsonl"
    identity = {"config": cfg, "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Existing evaluation has different inputs")
    manifest.write_text(json.dumps(identity, indent=2))
    results = (
        [json.loads(line) for line in results_path.read_text().splitlines()]
        if results_path.exists()
        else []
    )
    done = {row["task_id"] for row in results}
    for task in selected:
        if task["task_id"] in done:
            continue
        started = time.monotonic()
        request = {
            "model": cfg["model"],
            "temperature": 0,
            "seed": cfg.get("seed", 42),
            "max_tokens": cfg.get("max_tokens", 1024),
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": "Implement this Python function. Return the complete function, including its signature "
                    "and any imports, as Python code only. Do not include tests.\n\n"
                    + task["prompt"],
                }
            ],
        }
        row = {"task_id": task["task_id"], "passed": False}
        try:
            with urlopen(
                Request(
                    cfg["endpoint"] + "/v1/chat/completions",
                    data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=cfg.get("request_timeout", 300),
            ) as stream:
                response = json.load(stream)
            response_text = json.dumps(response, indent=2)
            (output / (task["task_id"].replace("/", "_") + ".json")).write_text(response_text)
            choice = response["choices"][0]
            code = choice["message"].get("content") or ""
            blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            if blocks:
                code = blocks[-1]
            row.update(
                finish_reason=choice.get("finish_reason"),
                usage=response.get("usage"),
                response_sha256=hashlib.sha256(response_text.encode()).hexdigest(),
            )
            row.update(execute(code, task, cfg["sandbox_image"]))
        except (OSError, ValueError, KeyError, IndexError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.monotonic() - started
        with results_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        results.append(row)
        report = {
            "kind": "sampled_coding_diagnostic",
            "samples": len(results),
            "passed": sum(r["passed"] for r in results),
            "pass_fraction": sum(r["passed"] for r in results) / len(results),
            "complete": len(results) == len(selected),
            "release_approved": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
