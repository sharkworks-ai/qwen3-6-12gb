"""Compile guided answers into a bounded, reproducible run plan."""

from __future__ import annotations

import os
import re
from pathlib import Path

from appliance.proof.config import defaults, validation_defaults
from appliance.stages.common import data_path, load_config


def harness_profiles():
    # Operator-owned configuration. The web client cannot supply URLs or commands.
    path = os.environ.get("QWEN12G_BENCHMARK_PROFILES")
    profiles = load_config(path) if path else {}
    return profiles


def integer(answers, key, default, low, high):
    value = int(answers.get(key, default))
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def compile_plan(answers, root="/data", *, check_inputs=True):
    if not isinstance(answers, dict):
        raise TypeError("Wizard answers must be an object")
    goal = answers.get("goal", "proof")
    if goal not in {"proof", "train", "compress"}:
        raise ValueError("Choose a proof, training or compression run")
    name = str(answers.get("name", "my-first-run"))
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Run name must contain letters, numbers, hyphens or underscores")
    devices = str(answers.get("cuda_devices", "0"))
    if not re.fullmatch(r"\d+(,\d+)*", devices) or len(set(devices.split(","))) != len(
        devices.split(",")
    ):
        raise ValueError("Select distinct numeric GPU indices, for example 0 or 0,1")
    count = len(devices.split(","))
    output = data_path(f"artifacts/wizard/{name}", root)
    resume = answers.get("resume", False) is True
    limit = float(answers.get("vram_limit_gib", 11.5))
    if not 0 < limit <= 12:
        raise ValueError("Target memory limit must be between 0 and 12 GiB")
    common = {
        "output_dir": str(output),
        "cuda_devices": devices,
        "device": "cuda:0",
        "resume": resume,
    }
    if goal == "proof":
        size = str(answers.get("proof_size", "standard"))
        if size not in {"laptop", "standard"}:
            raise ValueError("Choose laptop or standard proof size")
        if count != 1:
            raise ValueError("Miniature proof uses one GPU")
        cfg = {
            **defaults(),
            **common,
            "layers": 5 if size == "laptop" else defaults()["layers"],
            "sequence_length": 64 if size == "laptop" else defaults()["sequence_length"],
            "samples": 2 if size == "laptop" else defaults()["samples"],
            "vram_limit_gib": limit,
            "train_steps": integer(answers, "train_steps", 20, 1, 100000),
            "reconstruct_steps": integer(answers, "quant_steps", 10, 2, 10000),
            "qat_steps": integer(answers, "qat_steps", 10, 1, 100000),
        }
        return {
            "schema_version": 1,
            "goal": goal,
            "name": name,
            "config": cfg,
            "stages": [
                "Build miniature model",
                "Train",
                "Baseline diagnostics",
                "B + C reconstruction",
                "QAT recovery",
                "Re-quantize",
                "Packed parity and memory checks",
            ],
            "notes": [
                "Synthetic proof only. Published benchmarks and full context are not tested.",
                "Memory fit is measured during the run, not guaranteed by this plan.",
            ],
        }
    variants = answers.get("variants", ["aggressive", "extreme"])
    if (
        not isinstance(variants, list)
        or not variants
        or len(set(variants)) != len(variants)
        or any(v not in {"aggressive", "extreme"} for v in variants)
    ):
        raise ValueError("Select B, C or both")
    source = data_path(str(answers.get("source_model", "")), root)
    if source == Path(root).resolve():
        raise ValueError("Choose a local model checkpoint")
    paths = {}
    required = ["calibration_file", "recovery_dataset", "heldout_file"]
    if goal == "train":
        required.append("training_file")
    if "extreme" in variants:
        required.append("prompts_file")
    for key in required:
        if not answers.get(key):
            raise ValueError(f"Choose {key}")
        paths[key] = str(data_path(str(answers[key]), root))
        if check_inputs and not Path(paths[key]).is_file():
            raise ValueError(f"Missing input file: {key}")
    if check_inputs:
        model = load_config(source / "config.json")
        if model.get("qwen12g_proof_model"):
            raise ValueError("Use the proof goal for synthetic models")
        text = model.get("text_config", model)
        if int(text.get("max_position_embeddings", 0)) < 262144:
            raise ValueError("Source must support the native 262144 context")
    # Inputs can never be rewritten by a stage or mixed into held-out evaluation.
    for path in [source, *(Path(v) for v in paths.values())]:
        if path == output or output in path.parents or path in output.parents:
            raise ValueError("Inputs and output directory must be separate")
    heldout = Path(paths["heldout_file"])
    if any(heldout == Path(v) for k, v in paths.items() if k != "heldout_file"):
        raise ValueError("Held-out data must be separate from training and calibration")
    context = integer(answers, "context_length", 32768, 256, 250000)
    generated = integer(answers, "max_new_tokens", 8192, 1, 16384)
    if context + generated > 262144:
        raise ValueError("Prompt plus output exceeds native context")
    profile = str(answers.get("benchmark_profile", "diagnostics"))
    if profile != "diagnostics" and profile not in harness_profiles():
        raise ValueError("Selected isolated benchmark harness is not configured")
    keep = integer(answers, "keep_experts", 128, 1, 4096) if answers.get("prune") is True else None
    if (
        keep is not None
        and check_inputs
        and not int(text.get("num_experts_per_tok", 8)) <= keep < int(text.get("num_experts", 0))
    ):
        raise ValueError("Retained experts must be below source count and at least top-k")
    cfg = {
        **validation_defaults(),
        **common,
        **paths,
        "source_model": str(source),
        "variants": variants,
        "goal": goal,
        "load_in_4bit": answers.get("load_in_4bit", os.environ.get("QWEN12G_GPU_BACKEND") != "rocm") is True,
        "keep_experts": keep,
        "benchmark_profile": profile,
        "vram_limit_gib": limit,
        "steps": integer(answers, "quant_steps", 100, 2, 10000),
        "max_samples": integer(answers, "samples", 128, 1, 10000),
        "calibration_length": integer(answers, "sequence_length", 2048, 128, 16384),
        "train_steps": integer(answers, "train_steps", 200, 1, 100000),
        "train_length": integer(answers, "train_length", 4096, 128, 32768),
        "context_lengths": [context],
        "context_repeats": integer(answers, "context_repeats", 5, 1, 20),
        "max_new_tokens": generated,
        "runtime_device": "cuda:0",
        "chunk_rows": 128,
        "window_devices": [f"cuda:{i}" for i in range(count)],
        "max_memory": {
            **{
                str(i): f"{integer(answers, 'gpu_memory_gib', 28, 4, 192)}GiB" for i in range(count)
            },
            "cpu": f"{integer(answers, 'cpu_memory_gib', 80, 4, 1024)}GiB",
        },
        "recovery": {
            "num_processes": count,
            "max_steps": integer(answers, "qat_steps", 200, 1, 100000),
            "save_steps": 50,
        },
    }
    steps = ["SFT / QLoRA", "Merge adapter"] if goal == "train" else []
    steps += ["Profile expert use", "Prune experts"] if keep is not None else []
    steps += ["Generate teacher reasoning calibration"] if "extreme" in variants else []
    steps += [
        "Baseline diagnostics",
        "Selected B/C quantization",
        "QAT recovery",
        "Re-quantize",
        "Packed held-out diagnostics",
        "Repeated context and memory benchmarks",
    ]
    if profile != "diagnostics":
        steps += ["Isolated published benchmarks", "OxCoder comparison"]
    steps += ["Rank candidates and save report"]
    return {
        "schema_version": 1,
        "goal": goal,
        "name": name,
        "config": cfg,
        "stages": steps,
        "notes": [
            "The eager packed runtime has no fused production kernels or quantized KV cache.",
            "Diagnostics do not establish coding/tool quality. Published suites require a configured isolated harness.",
            "Failed gates remain visible. This workflow does not publish or approve a release.",
        ],
    }
