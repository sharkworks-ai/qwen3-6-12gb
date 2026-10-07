"""Web-launchable packed text generation, with no tool/command execution."""

from __future__ import annotations

import argparse
import os
import time

import torch

from appliance.runtime.packed import load_packed, sha256
from appliance.stages.common import data_path, load_config, save_json


@torch.inference_mode()
def run(cfg):
    from transformers import AutoTokenizer

    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    bundle = data_path(cfg["bundle"], root)
    output = data_path(cfg["output_dir"], root)
    if bundle == output or bundle in output.parents or output in bundle.parents:
        raise ValueError("Runtime output must be separate from its artifact")
    device = cfg.get("device", "cuda:0")
    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("Packed inference job requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = load_packed(bundle, device=device, chunk_rows=int(cfg.get("chunk_rows", 128)))
    model.set_attn_implementation("sdpa")
    tokenizer = AutoTokenizer.from_pretrained(bundle / "research-hf", trust_remote_code=False)
    tokens = tokenizer(str(cfg["prompt"]), return_tensors="pt").to(device)
    maximum = int(cfg.get("max_new_tokens", 128))
    if maximum < 1 or maximum + tokens.input_ids.shape[-1] > 262144:
        raise ValueError("Invalid prompt plus generation budget")
    loaded_seconds = time.monotonic() - started
    started = time.monotonic()
    sequence = model.generate(
        **tokens, max_new_tokens=maximum, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    count = sequence.shape[-1] - tokens.input_ids.shape[-1]
    result = {
        "runtime": "packed_eager",
        "production_runtime": False,
        "artifact_manifest_sha256": sha256(bundle / "mixed-manifest.json"),
        "config": cfg,
        "answer": tokenizer.decode(
            sequence[0, tokens.input_ids.shape[-1] :], skip_special_tokens=True
        ),
        "prompt_tokens": tokens.input_ids.shape[-1],
        "generated_tokens": count,
        "load_seconds": loaded_seconds,
        "generation_seconds": elapsed,
        "generation_tokens_per_second": count / max(elapsed, 1e-9),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "note": "Text generation only; no generated commands or tools are executed.",
    }
    save_json(output / "generation.json", result)
    print(result["answer"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
