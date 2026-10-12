"""One resumable job from guided intent to measured candidate comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from appliance.proof.state import Stages, hashes, implementation_digest
from appliance.quant.locking import exclusive
from appliance.quant.precision import digest
from appliance.runtime.packed import sha256
from appliance.stages.common import load_config, save_json
from appliance.wizard.config import compile_plan, harness_profiles, released_source


def remove_model_dir(path):
    """Delete a checkpoint directory, including cache blobs that only it links to.

    Snapshot files are symlinks; only the blobs free space. Another snapshot of the same
    Hugging Face repo may link the same blob, so keep any blob still referenced elsewhere.
    """
    path = Path(path).resolve()
    repo = next((p for p in path.parents if p.name.startswith("models--")), None)
    shared = set()
    if repo is not None:
        for other in (repo / "snapshots").glob("*"):
            if other.resolve() == path:
                continue
            for item in other.rglob("*"):
                if item.is_symlink():
                    shared.add(item.resolve())
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            target = item.resolve()
            if (
                repo is not None
                and repo in target.parents
                and target.is_file()
                and target not in shared
            ):
                target.unlink()
    shutil.rmtree(path)


ROLE_TURN = re.compile(
    r"(?:^|\n\n)user: (.*?)(?=\n\n(?:system|user|assistant|tool): |\Z)", re.DOTALL
)


def user_prompts(path, minimum=32):
    """Normalized task prompts: the first user turn of each conversation or text row.

    Later turns are often templated follow-ups ("return the final function again ...")
    shared by unrelated conversations, so comparing them flags boilerplate, not leakage.
    """
    found = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("messages"):
            texts = [m.get("content") or "" for m in row["messages"] if m.get("role") == "user"]
        else:
            texts = ROLE_TURN.findall(row.get("text") or "")
        if texts:
            normalized = " ".join(str(texts[0]).split()).lower()
            if len(normalized) >= minimum:
                found.add(normalized)
    return found


def heldout_overlap(config):
    """Count held-out user prompts that also appear in training, recovery or prompt data."""
    heldout = user_prompts(config["heldout_file"])
    return {
        key: len(heldout & user_prompts(config[key]))
        for key in ("training_file", "recovery_dataset", "prompts_file")
        if config.get(key)
    }


def release_source(stages, source):
    """Delete the base checkpoint after merge; resume reuses its recorded hashes."""
    stages.state.setdefault("released_inputs", {})["source_model"] = str(source)
    save_json(stages.path, stages.state)
    if source.exists():
        remove_model_dir(source)


def context_coverage(contexts, required_tokens):
    """Whether context checks reached the required length and sustained generation.

    Passing needle checks at short prompts, or a needle answer that stops early, says
    nothing about the full window or the generation budget, so both are required: a
    retrieval check at the required length and a forced-length run that generated its
    whole budget there.
    """
    retrieval = [c for c in contexts if not c.get("forced_length")]
    forced = [c for c in contexts if c.get("forced_length")]
    # Results without the evidence fields (older runs) count as not covered.
    reached = any(c.get("prompt_tokens", 0) >= required_tokens for c in retrieval)
    sustained = any(
        c.get("prompt_tokens", 0) >= required_tokens
        and c.get("generated_tokens", 0) >= c.get("max_new_tokens", float("inf"))
        for c in forced
    )
    return reached and sustained


def rank_candidates(validation, benchmarks=None):
    required = int(validation.get("required_context_tokens", 250000))
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
        retrieval = bool(contexts) and all(
            c["needle_found"] for c in contexts if not c.get("forced_length")
        )
        coverage = bool(contexts) and context_coverage(contexts, required)
        quality = bool(benchmarks) and benchmarks["candidates"][name]["gate"]["pass"]
        candidates.append(
            {
                "name": name,
                "heldout_loss": metrics["heldout_loss"],
                "bundle": variant["bundle"],
                "memory_pass": memory,
                "retrieval_pass": retrieval,
                "context_coverage_pass": coverage,
                "benchmark_pass": quality,
                "eligible": memory and retrieval and coverage and quality,
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
        source_path = Path(config["source_model"])
        if released_source(output, source_path):
            inputs["source_model"] = load_config(output / "input-manifest.json")["source_model"]
        else:
            inputs["source_model"] = hashes(source_path)
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
        # File hashes only catch whole-file copies; check individual prompts too.
        inputs["heldout_overlap"] = heldout_overlap(config)
        if any(inputs["heldout_overlap"].values()):
            raise ValueError(
                f"Held-out prompts also appear in other inputs: {inputs['heldout_overlap']}"
            )
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
            base = config["source_model"]
            source = base

            def measure(name, model):
                # Same held-out set and settings for every model, so losses compare
                # across pruning, SFT, quantization and QAT.
                return stages.run(
                    name,
                    "evaluate",
                    source_model=model,
                    heldout_file=config["heldout_file"],
                    hf_device_map="auto",
                    capture_logits=False,
                )

            lineage = {"original": measure("lineage-original", base)}
            # Prune before SFT: fused MoE experts stay BF16 under 4-bit loading, so the
            # full expert set does not fit 2x32 GB for training, while the pruned model
            # does. SFT then also recovers quality lost to pruning.
            if config["keep_experts"]:
                profile_path = output / "profile" / "expert-profile.json"
                stage(
                    "profile",
                    "profile",
                    {
                        "model": source,
                        # Inference only: BF16 may offload to CPU, which 4-bit forbids,
                        # and 4-bit would quantize only the small non-expert layers.
                        "load_in_4bit": False,
                        "max_memory": config["max_memory"],
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
                lineage["pruned"] = measure("lineage-pruned", str(pruned))
                source = str(pruned)
            if plan["goal"] == "train":
                adapter = output / "sft" / "adapter"
                training = {
                    "model": source,
                    "dataset_path": config["training_file"],
                    "output_dir": str(adapter),
                    "max_steps": config["train_steps"],
                    "max_length": config["train_length"],
                    # One process sharded across the selected GPUs.
                    "num_processes": 1,
                    "model_parallel": True,
                    "max_memory": config["max_memory"],
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
                if config["keep_experts"] and not config["keep_intermediates"]:
                    # Only SFT and merge read the pruned model.
                    stages.release("prune", "model")
                if config["release_source_after_merge"]:
                    release_source(stages, Path(base))
                source = str(merged)
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
            validation["required_context_tokens"] = config["required_context_tokens"]
            # Validation's baseline is the model compressed: SFT-merged if trained.
            lineage["sft" if plan["goal"] == "train" else "compression_source"] = validation.get(
                "baseline"
            )
            lineage["variants"] = {
                name: variant.get("lineage", {}) for name, variant in validation["variants"].items()
            }
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
                # Held-out loss at every step from the original model to each final
                # bundle, so total loss is visible, not just loss after quantization.
                "lineage": lineage,
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
