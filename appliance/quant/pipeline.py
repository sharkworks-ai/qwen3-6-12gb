"""Single registered job coordinating reconstruction, QAT and requantization."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
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
    # Marker first: if deletion is interrupted, resume still skips the first pass and
    # this call (idempotent) finishes the deletion.
    save_json(out / RELEASED, {"released": parts})
    for part in parts:
        shutil.rmtree(out / part, ignore_errors=True)


def measure(spec, model, result: Path):
    """Held-out loss of one model, cached so a resume never needs a released model."""
    if result.exists():
        return load_config(result)
    config = result.with_suffix(".config.json")
    save_json(config, spec)
    subprocess.run(
        [sys.executable, "-m", "appliance.proof.evaluate", "--config", str(config),
         "--source", str(model), "--output", str(result)],
        check=True,
    )
    return load_config(result)


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
        release_first_pass(first_dir)
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
    spec = cfg.get("lineage_eval")
    lineage = {}
    if spec:
        lineage["first_pass"] = measure(spec, first["research_model"], out / "lineage-first-pass.json")
    # QAT creates its own config identity + resumable Trainer checkpoints.
    subprocess.run(command_for("qat_recovery", qat_path, qat_config), check=True)
    if spec:
        lineage["qat"] = measure(spec, recovered, out / "lineage-qat.json")
    if not cfg.get("keep_intermediates"):
        release_first_pass(out)
    final_cfg = {**cfg, "source_model": str(recovered), "output_dir": str(out / "requantized")}
    # AYOT provenance remains tied to the original teacher during recovery.
    if cfg["preset"] == "extreme":
        final_cfg["teacher_model"] = cfg["source_model"]
    final = reconstruct(final_cfg)
    save_json(
        out / "pipeline-manifest.json",
        {
            "initial": first,
            "qat": str(out / "qat"),
            "final": final,
            "lineage": lineage,
            "runtime_ready": False,
        },
    )
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(json.dumps(run(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
