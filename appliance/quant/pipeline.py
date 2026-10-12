"""Single registered job coordinating reconstruction, QAT and requantization."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

from appliance.registry import command_for
from appliance.stages.common import data_path, load_config, save_json

RELEASED = "first-pass-released.json"


def release_first_pass(out: Path) -> None:
    """Delete the first-pass export once QAT has saved its recovered model.

    Only QAT reads it, and it is the largest output while re-quantization runs. The
    marker lets a resumed pipeline skip the first pass instead of re-verifying it.
    """
    parts = ["research-hf", "packed"]
    for part in parts:
        shutil.rmtree(out / part, ignore_errors=True)
    save_json(out / RELEASED, {"released": parts})


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
    first_dir = data_path(cfg["output_dir"], os.environ.get("QWEN12G_DATA_ROOT", "/data"))
    if (first_dir / RELEASED).exists():
        first = load_config(first_dir / "mixed-manifest.json")
    else:
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
    # Reconstruction ran in this process; release its cached GPU memory (about 9 GB on
    # cuda:0) so the QAT subprocess can use the whole device.
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # QAT creates its own config identity + resumable Trainer checkpoints.
    subprocess.run(command_for("qat_recovery", qat_path, qat_config), check=True)
    if not cfg.get("keep_intermediates"):
        release_first_pass(out)
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
