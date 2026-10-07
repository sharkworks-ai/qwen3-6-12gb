from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    description: str
    module: str
    gpu_required: bool = True
    distributed: bool = False


JOBS: dict[str, JobSpec] = {
    "upstream_reconstruction": JobSpec("upstream_reconstruction", "Pinned GSQ / CAT-Q sliding reconstruction", "GPTQ-initialized upstream GSQ with CAT-Q grid adaptation and overlapping window targets.", "appliance.quant.window_job"),
    "mixed_quant": JobSpec("mixed_quant", "Mixed GSQ / AYOT ternary", "Experimental B/C reconstruction with saved tensor precision maps.", "appliance.quant.mixed_job"),
    "mixed_pipeline": JobSpec("mixed_pipeline", "B/C compression pipeline", "Reconstruction, selective QAT recovery and re-quantization.", "appliance.quant.pipeline"),
    "ayot": JobSpec("ayot", "Teacher reasoning calibration", "Generate resumable source-model reasoning traces.", "appliance.quant.ayot"),
    "smoke": JobSpec("smoke", "GPU smoke test", "Validate CUDA and GPU telemetry.", "appliance.smoke"),
    "sft": JobSpec("sft", "SFT / QLoRA", "Agent/coding supervised fine-tuning with PEFT MoE expert adapters.", "appliance.stages.sft", distributed=True),
    "merge": JobSpec("merge", "Merge adapter", "Merge a recovered PEFT adapter into a full BF16 checkpoint for conversion/quantization.", "appliance.stages.merge"),
    "profile": JobSpec("profile", "Expert profile", "Collect per-layer expert routing frequency and routing mass.", "appliance.stages.profile"),
    "prune": JobSpec("prune", "Expert prune", "Create a workload-selected uniformly pruned Qwen3.6 checkpoint.", "appliance.stages.prune", gpu_required=False),
    "calibration": JobSpec("calibration", "Calibration corpus", "Materialize representative chat/tool text for importance-matrix calibration.", "appliance.stages.calibration", gpu_required=False),
    "quantize": JobSpec("quantize", "GGUF quantize", "Convert to GGUF, optionally build an imatrix, and apply mixed tensor quantization.", "appliance.stages.quantize"),
    "post_quant_recovery": JobSpec("post_quant_recovery", "Post-quant recovery", "Teacher-failure replay recovery using the SFT/QLoRA stage.", "appliance.recovery.post_quant", distributed=True),
    "qat_recovery": JobSpec("qat_recovery", "QAT recovery", "Selective fake-quant recovery for Q3/Q2/ternary compression.", "appliance.qat.job", distributed=True),
}


def get_job(name: str) -> JobSpec:
    try:
        return JOBS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registered job: {name}") from exc


def command_for(kind: str, config_path: Path, config: dict[str, Any]) -> list[str]:
    spec = get_job(kind)
    if kind == "smoke":
        return [sys.executable, "-m", spec.module]
    base = ["-m", spec.module, "--config", str(config_path)]
    processes = int(config.get("num_processes", 1))
    if spec.distributed and processes > 1:
        return ["torchrun", "--standalone", f"--nproc-per-node={processes}", *base]
    return [sys.executable, *base]
