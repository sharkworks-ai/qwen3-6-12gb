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

# First-pass reconstruction outputs and QAT state; the requantized bundle is kept.
RELEASED_PIPELINE_OUTPUTS = ("research-hf", "packed", "training-checkpoints", "qat")


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
    tokens, secret, position = needle_prompt(tokenizer, length, int(cfg.get("repetition", 0)))
    inputs = torch.tensor([tokens], device=cfg["runtime_device"])
    torch.cuda.reset_peak_memory_stats(cfg["runtime_device"])
    forced = bool(cfg.get("forced_length"))
    output = model.generate(
        inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=generated,
        # Forced length: an early EOS would leave the generation budget unmeasured.
        min_new_tokens=generated if forced else None,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(output[0, length:], skip_special_tokens=True)
    return {
        "prompt_tokens": length,
        "generated_tokens": output.shape[-1] - length,
        "max_new_tokens": generated,
        "forced_length": forced,
        "needle_position": position,
        "needle_found": secret in answer,
        "answer": answer[:2000],
        "peak_allocated_mib": torch.cuda.max_memory_allocated(cfg["runtime_device"]) / 1024**2,
        "note": "Needle retrieval with distractors, plus a forced-length memory run; "
        "not repository-scale coding or tool-history validation.",
    }


NEEDLE_POSITIONS = (0.1, 0.5, 0.9, 0.3, 0.7)


def needle_prompt(tokenizer, length, repetition):
    """Build an exact-length prompt with one secret code among decoy codes.

    Each repetition moves the needle and changes the secret, so repeats test retrieval
    across the window rather than re-running one identical prompt.
    """
    import random

    rng = random.Random(1000 + repetition)
    colours = ["BLUE", "RED", "GREEN", "AMBER", "VIOLET", "SILVER", "ORANGE", "TEAL"]

    def code():
        return f"{rng.choice(colours)}-{rng.randrange(1000, 10000)}"

    secret = code()

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    # A chat-tuned model given raw text ends its turn at once, so ask as a user turn
    # with the reply opened (thinking off) whenever the tokenizer has a chat template.
    head, tail = "", ""
    if getattr(tokenizer, "chat_template", None):
        marker = "\x00"
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        head, tail = wrapped.split(marker)
    prefix = encode(head + "Remember the secret code in these notes. Old codes are retired.\n")
    needle = encode(f"\nThe current secret code is {secret}.\n")
    suffix = encode("\nReturn only the current secret code." + tail)
    filler = encode("Ordinary project notes. No secret code here.\n")
    decoys = [encode(f"\nA retired code was {code()}; it is no longer valid.\n") for _ in range(4)]
    fixed = len(prefix) + len(needle) + len(suffix) + sum(len(d) for d in decoys)
    if not filler or length <= fixed + len(filler):
        raise ValueError("Invalid context length or tokenizer")
    available = length - fixed
    body = (filler * (available // len(filler) + 1))[:available]
    position = NEEDLE_POSITIONS[repetition % len(NEEDLE_POSITIONS)]
    inserts = sorted(
        [(int(available * position), needle)]
        + [(int(available * (i + 1) / (len(decoys) + 1)), d) for i, d in enumerate(decoys)],
        key=lambda item: item[0],
        reverse=True,
    )
    for at, piece in inserts:
        body = body[:at] + piece + body[at:]
    tokens = prefix + body + suffix
    if len(tokens) != length:
        raise ValueError("Needle prompt length mismatch")
    return tokens, secret, position


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
            "recovery_dataset",
            "heldout_file",
        )
    }
    names = cfg.get("variants", ["aggressive", "extreme"])
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in {"aggressive", "extreme"} for n in names)
    ):
        raise ValueError("Choose aggressive, extreme or both")
    if "extreme" in names:
        paths["ayot_file"] = data_path(cfg["ayot_file"], root)
    for name, path in paths.items():
        if not (path.is_dir() if name == "source_model" else path.is_file()):
            raise ValueError(f"Missing validation input: {name}")
    if load_config(paths["source_model"] / "config.json").get("qwen12g_proof_model"):
        raise ValueError(
            "Full validation requires a trained real checkpoint, not a synthetic proof model"
        )
    if any(
        paths["heldout_file"] == path
        for name, path in paths.items()
        if name not in {"heldout_file", "source_model"}
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
    provenance = load_components(
        {**cfg, "preset": "extreme" if "extreme" in names else "aggressive"}
    )[3]
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
        for name in names:
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
                "keep_intermediates": bool(cfg.get("keep_intermediates")),
                "steps": cfg["steps"],
                "max_samples": cfg["max_samples"],
                "calibration_length": cfg["calibration_length"],
                "recovery": {
                    **preset(name)["recovery"],
                    **cfg["recovery"],
                    "enabled": True,
                    "dataset": str(paths["recovery_dataset"]),
                },
                # Measure the first pass and QAT output before the pipeline releases them.
                "lineage_eval": {
                    "heldout_file": str(paths["heldout_file"]),
                    "eval_samples": cfg["eval_samples"],
                    "eval_length": cfg["eval_length"],
                    "max_memory": cfg["max_memory"],
                    "device": cfg["device"],
                    "chunk_rows": cfg["chunk_rows"],
                    "hf_device_map": "auto",
                },
            }
            stages.run(f"{name}-pipeline", "pipeline", quant_config=quant)
            manifest = directory / "pipeline-manifest.json"
            lineage = load_config(manifest).get("lineage", {}) if manifest.exists() else {}
            if not cfg.get("keep_intermediates"):
                # Requantization succeeded, so the first-pass export and QAT state are
                # superseded; drop them before the next variant starts.
                for part in RELEASED_PIPELINE_OUTPUTS:
                    stages.release(f"{name}-pipeline", f"artifact/{part}")
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
            physical = str(cfg["cuda_devices"]).split(",")[
                int(cfg["runtime_device"].split(":")[-1])
            ]

            def context(stage, final=final, physical=physical, contexts=contexts, **options):
                result = stages.run(stage, "context", bundle=str(final), **options)
                result["peak_device_mib"] = stages.state["stages"][stage]["peak_vram_mib"].get(
                    physical
                )
                contexts.append(result)

            for length in cfg["context_lengths"]:
                for repetition in range(int(cfg["context_repeats"])):
                    context(
                        f"{name}-context-{length}-{repetition}",
                        context_length=length,
                        repetition=repetition,
                    )
            # Needle answers stop early; prove the whole generation budget fits at the
            # longest prompt with a run that cannot stop before max_new_tokens.
            context(
                f"{name}-forced-{max(cfg['context_lengths'])}",
                context_length=max(cfg["context_lengths"]),
                forced_length=True,
            )
            variants[name] = {
                "metrics": metrics,
                "contexts": contexts,
                "bundle": str(final),
                "lineage": {**lineage, "final_packed": metrics},
            }
        from appliance.wizard.job import context_coverage

        required_context = int(cfg.get("required_context_tokens", 250000))
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
            "required_context_tokens": required_context,
            "identity": identity,
            "variants": variants,
            "baseline": baseline,
            "stages": stages.state["stages"],
            "benchmark_gate": benchmark_gate,
            "benchmark_execution": "external_isolated_harness_required",
            "checks": {
                "needle_retrieval": all(
                    c["needle_found"]
                    for v in variants.values()
                    for c in v["contexts"]
                    if not c.get("forced_length")
                ),
                "context_coverage": all(
                    context_coverage(v["contexts"], required_context)
                    for v in variants.values()
                ),
                "within_vram_limit": all(
                    c.get("peak_device_mib") is not None
                    and max(c["peak_allocated_mib"], c["peak_device_mib"]) / 1024
                    <= float(cfg["vram_limit_gib"])
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
