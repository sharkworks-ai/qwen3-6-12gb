"""Full-model compression, packed runtime checks and explicit release gates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from appliance.eval.oxcoder_gate import BASELINE, score
from appliance.proof.config import validation_defaults
from appliance.proof.state import Stages, implementation_digest
from appliance.quant.locking import exclusive
from appliance.quant.precision import digest
from appliance.quant.presets import preset
from appliance.quant.upstream import load_components
from appliance.runtime.packed import load_packed, sha256
from appliance.stages.common import data_path, load_config, save_json


@torch.inference_mode()
def context_check(cfg):
    from transformers import AutoTokenizer

    bundle = Path(cfg["bundle"])
    tokenizer = AutoTokenizer.from_pretrained(bundle / "research-hf", trust_remote_code=False)
    model = load_packed(bundle, device=cfg["runtime_device"], chunk_rows=int(cfg["chunk_rows"]))
    model.set_attn_implementation("sdpa")
    length = int(cfg["context_length"])
    generated = int(cfg["max_new_tokens"])
    if length + generated > 262144:
        raise ValueError("Prompt plus generation exceeds native context")
    prefix = tokenizer.encode(
        "Remember the secret code in these notes.\n", add_special_tokens=False
    )
    needle = tokenizer.encode("\nThe secret code is BLUE-7319.\n", add_special_tokens=False)
    suffix = tokenizer.encode("\nReturn only the secret code.\n", add_special_tokens=False)
    filler = tokenizer.encode(
        "Ordinary project notes. No secret code here.\n", add_special_tokens=False
    )
    if not filler or length <= len(prefix) + len(needle) + len(suffix):
        raise ValueError("Invalid context length or tokenizer")
    available = length - len(prefix) - len(needle) - len(suffix)
    repeated = (filler * (available // len(filler) + 1))[:available]
    tokens = prefix + repeated[: available // 2] + needle + repeated[available // 2 :] + suffix
    inputs = torch.tensor([tokens], device=cfg["runtime_device"])
    torch.cuda.reset_peak_memory_stats(cfg["runtime_device"])
    output = model.generate(
        inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=generated,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(output[0, length:], skip_special_tokens=True)
    return {
        "prompt_tokens": length,
        "generated_tokens": output.shape[-1] - length,
        "needle_found": "BLUE-7319" in answer,
        "answer": answer,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(cfg["runtime_device"]) / 1024**2,
        "note": "Needle retrieval diagnostic; not full-history coding validation.",
    }


def run(cfg):
    cfg = {**validation_defaults(), **cfg}
    cfg.setdefault("chunk_rows", 128)
    cfg.setdefault("seed", 42)
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    output = data_path(cfg["output_dir"], root)
    paths = {
        name: data_path(cfg[name], root)
        for name in (
            "source_model",
            "calibration_file",
            "ayot_file",
            "recovery_dataset",
            "heldout_file",
        )
    }
    for name, path in paths.items():
        if not (path.is_dir() if name == "source_model" else path.is_file()):
            raise ValueError(f"Missing validation input: {name}")
    if load_config(paths["source_model"] / "config.json").get("qwen12g_proof_model"):
        raise ValueError(
            "Full validation requires a trained real checkpoint, not a synthetic proof model"
        )
    if any(
        paths["heldout_file"] == paths[name]
        for name in ("calibration_file", "ayot_file", "recovery_dataset")
    ):
        raise ValueError("Held-out data must be separate from calibration and recovery")
    if not cfg["context_lengths"] or int(cfg["context_repeats"]) < 1:
        raise ValueError("Specify context lengths and positive repeat count")
    if any(not 1 <= int(n) <= 262144 - int(cfg["max_new_tokens"]) for n in cfg["context_lengths"]):
        raise ValueError("Invalid prompt plus generation context budget")
    if (
        output == paths["source_model"]
        or output in paths["source_model"].parents
        or paths["source_model"] in output.parents
    ):
        raise ValueError("Source and output must be separate")
    provenance = load_components({**cfg, "preset": "extreme"})[3]
    # Freeze real input evidence, not only path strings.
    inputs = {
        str(p): sha256(p)
        for path in paths.values()
        for p in (sorted(path.rglob("*")) if path.is_dir() else [path])
        if p.is_file()
    }
    benchmark = None
    if cfg.get("benchmark_results"):
        benchmark_path = data_path(cfg["benchmark_results"], root)
        benchmark = load_config(benchmark_path)
        inputs[str(benchmark_path)] = sha256(benchmark_path)
    identity = digest(
        {
            "config": {k: v for k, v in cfg.items() if k != "resume"},
            "inputs": inputs,
            "upstream": provenance,
            "implementation": implementation_digest(),
        }
    )
    with exclusive(output / ".validation.lock"):
        stages = Stages(cfg, output, identity)
        baseline = stages.run(
            "baseline",
            "evaluate",
            source_model=str(paths["source_model"]),
            heldout_file=str(paths["heldout_file"]),
            hf_device_map="auto",
            capture_logits=False,
        )
        variants = {}
        for name in ("aggressive", "extreme"):
            directory = output / f"{name}-pipeline" / "artifact"
            quant = {
                **preset(name),
                "source_model": str(paths["source_model"]),
                "output_dir": str(directory),
                "calibration_file": str(
                    paths["ayot_file"] if name == "extreme" else paths["calibration_file"]
                ),
                "device": cfg["device"],
                "window_devices": cfg["window_devices"],
                "resume": cfg["resume"],
                "dry_run": False,
                "steps": cfg["steps"],
                "max_samples": cfg["max_samples"],
                "calibration_length": cfg["calibration_length"],
                "recovery": {
                    **preset(name)["recovery"],
                    **cfg["recovery"],
                    "enabled": True,
                    "dataset": str(paths["recovery_dataset"]),
                },
            }
            stages.run(f"{name}-pipeline", "pipeline", quant_config=quant)
            final = directory / "requantized"
            metrics = stages.run(
                f"{name}-packed",
                "evaluate",
                source_model=str(final / "research-hf"),
                bundle=str(final),
                device=cfg["runtime_device"],
                heldout_file=str(paths["heldout_file"]),
                compare_export=False,
                capture_logits=False,
                baseline=str(output / "baseline" / "logits.pt"),
            )
            contexts = []
            for length in cfg["context_lengths"]:
                for repetition in range(int(cfg["context_repeats"])):
                    result = stages.run(
                        f"{name}-context-{length}-{repetition}",
                        "context",
                        bundle=str(final),
                        context_length=length,
                    )
                    contexts.append(result)
            variants[name] = {"metrics": metrics, "contexts": contexts, "bundle": str(final)}
        benchmark_gate = None
        if benchmark is not None:
            # Each candidate must supply its own scores and matching artifact identity.
            benchmark_gate = {}
            for name, variant in variants.items():
                candidate = benchmark.get(name, {})
                manifest_sha = sha256(Path(variant["bundle"]) / "mixed-manifest.json")
                if candidate.get("artifact_manifest_sha256") != manifest_sha or not candidate.get(
                    "harness_revisions"
                ):
                    raise ValueError(f"Benchmark provenance mismatch: {name}")
                scores = candidate.get("scores", {})
                if set(scores) != set(BASELINE) or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= 100 for v in scores.values()
                ):
                    raise ValueError("Require all 11 benchmark scores on a 0–100 scale")
                benchmark_gate[name] = score(scores)
        report = {
            "status": "completed",
            "identity": identity,
            "variants": variants,
            "baseline": baseline,
            "stages": stages.state["stages"],
            "benchmark_gate": benchmark_gate,
            "benchmark_execution": "external_isolated_harness_required",
            "checks": {
                "needle_retrieval": all(
                    c["needle_found"] for v in variants.values() for c in v["contexts"]
                ),
                "within_vram_limit": all(
                    c["peak_allocated_mib"] / 1024 <= float(cfg["vram_limit_gib"])
                    for v in variants.values()
                    for c in v["contexts"]
                ),
                "coding_tool_quality": False,
            },
            "production_runtime": False,
            "release_approved": False,
            "note": "Packed eager runtime diagnostics. Benchmark imports require provenance. A release requires isolated coding/tool loops and full-context quality retention.",
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
