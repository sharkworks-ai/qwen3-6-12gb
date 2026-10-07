"""One resumable job from guided intent to measured candidate comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from appliance.proof.state import Stages, hashes, implementation_digest
from appliance.quant.locking import exclusive
from appliance.quant.precision import digest
from appliance.runtime.packed import sha256
from appliance.stages.common import load_config, save_json
from appliance.wizard.config import compile_plan, harness_profiles


def rank_candidates(validation, benchmarks=None):
    candidates = []
    for name, variant in validation["variants"].items():
        metrics = variant.get("metrics", variant.get("after_qat", {}))
        contexts = variant.get("contexts", [])
        memory = bool(contexts) and all(
            c.get("peak_device_mib") is not None
            and max(c["peak_allocated_mib"], c["peak_device_mib"]) / 1024
            <= validation["target_vram_gib"]
            for c in contexts
        )
        retrieval = bool(contexts) and all(c["needle_found"] for c in contexts)
        quality = bool(benchmarks) and benchmarks["candidates"][name]["gate"]["pass"]
        candidates.append(
            {
                "name": name,
                "heldout_loss": metrics["heldout_loss"],
                "bundle": variant["bundle"],
                "memory_pass": memory,
                "retrieval_pass": retrieval,
                "benchmark_pass": quality,
                "eligible": memory and retrieval and quality,
            }
        )
    candidates.sort(key=lambda c: (not c["eligible"], c["heldout_loss"]))
    return {
        "candidates": candidates,
        "recommended": next((c["name"] for c in candidates if c["eligible"]), None),
        "release_approved": False,
        "ranking": "Gate-qualified candidates first; held-out loss breaks ties. Diagnostics alone cannot recommend a release.",
    }


def run(cfg):
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    plan = compile_plan(cfg["answers"], root)
    config = plan["config"]
    output = Path(config["output_dir"])
    inputs = {}
    if plan["goal"] != "proof":
        inputs["source_model"] = hashes(Path(config["source_model"]))
        inputs.update(
            {
                k: sha256(Path(v))
                for k, v in config.items()
                if k
                in {
                    "training_file",
                    "calibration_file",
                    "recovery_dataset",
                    "heldout_file",
                    "prompts_file",
                }
            }
        )
        if any(
            inputs["heldout_file"] == v
            for k, v in inputs.items()
            if k not in {"heldout_file", "source_model"}
        ):
            raise ValueError("Held-out input duplicates training/calibration data")
    profile = harness_profiles().get(config.get("benchmark_profile"))
    identity = digest(
        {
            "plan": {**plan, "config": {k: v for k, v in config.items() if k != "resume"}},
            "inputs": inputs,
            "implementation": implementation_digest(),
            "harness": profile,
        }
    )
    with exclusive(output / ".wizard.lock"):
        stages = Stages(config, output, identity)
        save_json(output / "plan.json", plan)
        save_json(output / "input-manifest.json", inputs)

        def stage(name, kind, job, result=None):
            return stages.run(
                name,
                "registered",
                job_kind=kind,
                job_config=job,
                result_file=str(result) if result else None,
            )

        if plan["goal"] == "proof":
            proof_cfg = {**config, "output_dir": str(output / "miniature-proof" / "artifact")}
            validation = stage(
                "miniature-proof",
                "proof_run",
                proof_cfg,
                output / "miniature-proof" / "artifact" / "report.json",
            )
            report = {**validation, "wizard_plan": plan, "pipeline_stages": stages.state["stages"]}
        else:
            source = config["source_model"]
            if plan["goal"] == "train":
                adapter = output / "sft" / "adapter"
                training = {
                    "model": source,
                    "dataset_path": config["training_file"],
                    "output_dir": str(adapter),
                    "max_steps": config["train_steps"],
                    "max_length": config["train_length"],
                    "num_processes": len(config["window_devices"]),
                    "resume": config["resume"],
                    "save_steps": 50,
                    "load_in_4bit": config.get("load_in_4bit", True),
                    "trust_remote_code": False,
                }
                stage("sft", "sft", training)
                merged = output / "merge" / "model"
                stage(
                    "merge",
                    "merge",
                    {"model": source, "adapter": str(adapter), "output_dir": str(merged)},
                )
                source = str(merged)
            if config["keep_experts"]:
                profile_path = output / "profile" / "expert-profile.json"
                stage(
                    "profile",
                    "profile",
                    {
                        "model": source,
                        "load_in_4bit": config.get("load_in_4bit", True),
                        "dataset_path": config["calibration_file"],
                        "output": str(profile_path),
                        "max_samples": config["max_samples"],
                        "max_length": config["calibration_length"],
                    },
                )
                pruned = output / "prune" / "model"
                stage(
                    "prune",
                    "prune",
                    {
                        "model": source,
                        "profile": str(profile_path),
                        "output_dir": str(pruned),
                        "keep_experts": config["keep_experts"],
                    },
                )
                source = str(pruned)
            if "extreme" in config["variants"]:
                ayot = output / "ayot" / "traces.jsonl"
                stage(
                    "ayot",
                    "ayot",
                    {
                        "source_model": source,
                        "prompts_file": config["prompts_file"],
                        "output_file": str(ayot),
                        "resume": config["resume"],
                        "max_samples": config["max_samples"],
                        "max_memory": config["max_memory"],
                    },
                )
            validation_cfg = {
                **config,
                "source_model": source,
                "output_dir": str(output / "validation" / "artifact"),
                "benchmark_results": "",
            }
            if "extreme" in config["variants"]:
                validation_cfg["ayot_file"] = str(ayot)
            validation = stage(
                "validation",
                "full_validation",
                validation_cfg,
                output / "validation" / "artifact" / "report.json",
            )
            validation["target_vram_gib"] = config["vram_limit_gib"]
            benchmarks = None
            if config["benchmark_profile"] != "diagnostics":
                benchmarks = stage(
                    "benchmarks",
                    "wizard_benchmarks",
                    {
                        "report_file": str(output / "validation" / "artifact" / "report.json"),
                        "benchmark_profile": config["benchmark_profile"],
                        "request_id": identity,
                        "output_dir": str(output / "benchmarks"),
                    },
                    output / "benchmarks" / "benchmark-report.json",
                )
            report = {
                **validation,
                "wizard_plan": plan,
                "pipeline_stages": stages.state["stages"],
                "benchmark_comparison": benchmarks,
                "selection": rank_candidates(validation, benchmarks),
            }
        save_json(output / "report.json", report)
        stages.state["status"] = "completed"
        save_json(stages.path, stages.state)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
