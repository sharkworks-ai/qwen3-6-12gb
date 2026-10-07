"""One-GPU, resumable baseline/B/C proof run."""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version

from appliance.proof.config import defaults, validate
from appliance.proof.state import Stages, implementation_digest
from appliance.quant.locking import exclusive
from appliance.quant.precision import digest
from appliance.quant.presets import preset
from appliance.quant.upstream import load_components
from appliance.stages.common import data_path, load_config, save_json


def run(cfg):
    cfg = {**defaults(), **cfg}
    validate(cfg)
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    output = data_path(cfg["output_dir"], root)
    with exclusive(output / ".proof.lock"):
        return _run(cfg, output)


def _run(cfg, output):
    provenance = load_components({**cfg, "preset": "extreme"})[3]
    identity = digest(
        {
            "config": {k: v for k, v in cfg.items() if k != "resume"},
            "upstream": provenance,
            "implementation": implementation_digest(),
            "accelerate": version("accelerate"),
        }
    )
    stages = Stages(cfg, output, identity)
    generated = stages.run("generate", "generate")
    data = output / "generate" / "datasets"
    original = output / "generate" / "source"
    trained = output / "bootstrap" / "model"
    stages.run(
        "bootstrap",
        "train",
        source_model=str(original),
        training_file=str(data / "train.jsonl"),
        training_steps=cfg["train_steps"],
    )
    common = {"heldout_file": str(data / "heldout.jsonl")}
    baseline = stages.run("baseline", "evaluate", source_model=str(trained), **common)
    variants = {}
    for name in ("aggressive", "extreme"):
        first = output / f"{name}-reconstruct" / "artifact"
        quant = {
            **preset(name),
            "source_model": str(trained),
            "output_dir": str(first),
            "calibration_file": str(data / "calibration.jsonl"),
            "proof_mode": True,
            "proof_interrupt_step": 1,
            "device": cfg["device"],
            "window_devices": [cfg["device"]],
            "group_size": 64,
            "calibration_length": cfg["sequence_length"],
            "max_samples": cfg["samples"],
            "steps": cfg["reconstruct_steps"],
            "checkpoint_steps": cfg["checkpoint_steps"],
            "dry_run": False,
            "resume": cfg["resume"],
            "seed": cfg["seed"],
        }
        # Checkpoint step 1 must be durable for the intentional restart check.
        quant["checkpoint_steps"] = 1
        for key in ("cpu_test", "gsq_root", "bittern_root"):
            if key in cfg:
                quant[key] = cfg[key]
        stages.run(f"{name}-reconstruct", "reconstruct", quant_config=quant)
        initial = stages.run(
            f"{name}-initial",
            "evaluate",
            source_model=str(first / "research-hf"),
            baseline=str(output / "baseline" / "logits.pt"),
            **common,
        )
        recovered = output / f"{name}-qat" / "model"
        stages.run(
            f"{name}-qat",
            "train",
            source_model=str(first / "research-hf"),
            precision_map=str(first / "precision-map.json"),
            training_file=str(data / "train.jsonl"),
            training_steps=cfg["qat_steps"],
        )
        final = output / f"{name}-requantize" / "artifact"
        final_quant = {**quant, "source_model": str(recovered), "output_dir": str(final)}
        stages.run(f"{name}-requantize", "reconstruct", quant_config=final_quant)
        runtime = stages.run(
            f"{name}-packed",
            "evaluate",
            source_model=str(final / "research-hf"),
            bundle=str(final),
            baseline=str(output / "baseline" / "logits.pt"),
            **common,
        )
        variants[name] = {
            "initial": initial,
            "after_qat": runtime,
            "bundle": str(final),
            "packed_bytes": sum(p.stat().st_size for p in (final / "packed").glob("*.safetensors")),
            "research_export_bytes": sum(
                p.stat().st_size for p in (final / "research-hf").glob("*.safetensors")
            ),
        }
    stage_items = stages.state["stages"]
    memory_values = [v for stage in stage_items.values() for v in stage["peak_vram_mib"].values()]
    measured = bool(memory_values)
    within_memory = measured and max(memory_values) / 1024 <= cfg["vram_limit_gib"]
    parity = all(v["after_qat"]["export_parity_pass"] for v in variants.values())
    report = {
        "status": "completed",
        "identity": identity,
        "model": generated,
        "baseline": baseline,
        "variants": variants,
        "stages": stage_items,
        "synthetic": True,
        "quality_validation": "not_tested",
        "ayot_validation": "not_tested_synthetic_fixture",
        "production_runtime": False,
        "full_context_validation": "not_tested",
        "checks": {
            "packed_export_parity": parity,
            "checkpoint_restart": all(
                stage_items[f"{n}-reconstruct"]["checkpoint_restart_checked"] for n in variants
            ),
            "vram_measured": measured,
            "within_vram_limit": within_memory,
        },
        "pass": parity
        and within_memory
        and all(
            stage_items[f"{name}-reconstruct"]["checkpoint_restart_checked"] for name in variants
        ),
        "note": "Synthetic miniature proof only. No coding, AYOT, full-model or long-context quality claim.",
    }
    save_json(output / "report.json", report)
    stages.state["status"] = "completed"
    save_json(stages.path, stages.state)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run(load_config(parser.parse_args().config))
    print(f"Proof report: {result['status']}; gates pass: {result['pass']}", flush=True)


if __name__ == "__main__":
    main()
