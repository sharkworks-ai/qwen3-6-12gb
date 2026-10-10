"""Single registered job coordinating reconstruction, QAT and requantization."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from appliance.registry import command_for
from appliance.stages.common import load_config, save_json


def run(cfg: dict):
    cfg = copy.deepcopy(cfg)
    recovery = cfg.pop("recovery", {})
    if cfg.get("engine", "reference") == "upstream_window":
        from appliance.quant.window_job import run as reconstruct
    else:
        from appliance.quant.mixed_job import run as reconstruct
    if cfg.get("preset") == "extreme" and not recovery.get("enabled", True):
        raise ValueError("Extreme pipeline requires QAT recovery")
    if recovery.get("enabled", cfg["preset"] == "extreme") and not recovery.get("dataset"):
        raise ValueError("QAT recovery dataset is required")
    first = reconstruct(cfg)
    if cfg.get("dry_run"):
        return first
    if not recovery.get("enabled", cfg["preset"] == "extreme"):
        return first
    if not recovery.get("dataset"):
        raise ValueError("QAT recovery dataset is required")
    out = Path(first["precision_map"]).parent
    recovered = out / "qat" / "recovered"
    qat_config = {
        **recovery,
        "student_model": first["research_model"],
        "precision_map": first["precision_map"],
        "output_dir": str(out / "qat"),
        "num_processes": int(recovery.get("num_processes", 1)),
        "resume": bool(cfg.get("resume")),
    }
    qat_path = out / "qat-config.json"
    save_json(qat_path, qat_config)
    # QAT creates its own config identity + resumable Trainer checkpoints.
    subprocess.run(command_for("qat_recovery", qat_path, qat_config), check=True)
    final_cfg = {**cfg, "source_model": str(recovered), "output_dir": str(out / "requantized")}
    # AYOT provenance remains tied to the original teacher during recovery.
    if cfg["preset"] == "extreme":
        final_cfg["teacher_model"] = cfg["source_model"]
    final = reconstruct(final_cfg)
    save_json(
        out / "pipeline-manifest.json",
        {"initial": first, "qat": str(out / "qat"), "final": final, "runtime_ready": False},
    )
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(json.dumps(run(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
