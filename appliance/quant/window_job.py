"""Mixed-precision block reconstruction using upstream GSQ and CAT-Q grids."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from torch.func import functional_call

from appliance.quant.locking import exclusive
from appliance.quant.mixed_job import file_hash, prepare
from appliance.quant.precision import build_map, digest
from appliance.quant.upstream import load_components
from appliance.quant.upstream_params import ProjectionQuantizer, schedule
from appliance.quant.window_cache import capture, find_blocks, hidden, move
from appliance.stages.common import data_path, load_config, save_json


def windows(total, width, stride):
    if not 1 <= stride <= width <= total:
        raise ValueError("Require 1 <= stride <= window_size <= decoder block count")
    start = 0
    while start < total:
        end = min(total, start + width)
        commit = end if end == total else start + stride
        yield start, end, commit
        start = commit


def run_blocks(blocks, quantizers, prefix, start, inputs, cache, sample, devices):
    value = inputs
    for offset, block in enumerate(blocks):
        index = start + offset
        device = devices[offset % len(devices)]
        call = torch.load(cache / str(index) / f"sample-{sample}.pt", weights_only=True)
        args = list(move(call["args"], device))
        kwargs = move(call["kwargs"], device)
        if args:
            args[0] = value.to(device)
        else:
            kwargs["hidden_states"] = value.to(device)
        weights = {
            name[len(f"{prefix}.{index}.") :]: q()
            for name, q in quantizers.items()
            if name.startswith(f"{prefix}.{index}.")
        }
        value = hidden(functional_call(block, weights, tuple(args), kwargs, strict=False))
    return value


def run(cfg):
    output = data_path(cfg["output_dir"], os.environ.get("QWEN12G_DATA_ROOT", "/data"))
    source = data_path(cfg["source_model"], os.environ.get("QWEN12G_DATA_ROOT", "/data"))
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories")
    with exclusive(output / ".window.lock"):
        return _run(cfg)


def _run(cfg):
    model_path, output, _files, plan = prepare(cfg)
    components = load_components(cfg)
    plan["upstream"] = components[3]
    plan["identity"] = digest({"input": plan["identity"], "upstream": components[3]})
    save_json(output / "window-plan.json", plan)
    if cfg.get("dry_run"):
        return {"status": "planned", "plan": str(output / "window-plan.json")}
    if str(cfg.get("device", "cuda:0")).startswith("cpu") and not cfg.get("cpu_test"):
        raise ValueError("GSQ's pinned autograd implementation requires CUDA")
    from lion_pytorch import Lion
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state_path = output / "window-progress.json"
    state = {"identity": plan["identity"], "next_block": 0, "completed": {}, "cache": {}}
    if state_path.exists():
        if not cfg.get("resume"):
            raise ValueError("Run exists; select Resume or a new output directory")
        state = load_config(state_path)
        if state["identity"] != plan["identity"]:
            raise ValueError("Window resume source/config/upstream mismatch")
        for name, item in state["completed"].items():
            if file_hash(output / item["file"]) != item["sha256"]:
                raise ValueError(f"Window artifact integrity failure: {name}")
        for path, value in state["cache"].items():
            if file_hash(output / path) != value:
                raise ValueError(f"Teacher cache integrity failure: {path}")
        for path, value in state.get("exports", {}).items():
            if file_hash(output / path) != value:
                raise ValueError(f"Committed reconstruction integrity failure: {path}")
        if (
            state.get("active")
            and file_hash(output / state["active"]["file"]) != state["active"]["sha256"]
        ):
            raise ValueError("Active window checkpoint integrity failure")
    if state.get("status") == "succeeded":
        return load_config(output / "mixed-manifest.json")
    torch.manual_seed(int(cfg.get("seed", 42)))
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    prefix, base_blocks = find_blocks(model)
    precision = build_map([n for n, _ in model.named_parameters()], cfg)
    # This reconstruction engine only targets matrix projections in decoder blocks.
    parameters = dict(model.named_parameters())
    precision["tensors"] = {
        n: s
        for n, s in precision["tensors"].items()
        if n.startswith(prefix + ".") and (parameters[n].ndim == 2 or ".experts." in n)
    }
    save_json(output / "precision-map.json", precision)
    corpus = data_path(cfg["calibration_file"], os.environ.get("QWEN12G_DATA_ROOT", "/data"))
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    rows = rows[: int(cfg.get("max_samples", 128))]
    if not rows:
        raise ValueError("Empty calibration corpus")
    cache = output / "teacher-cache"
    if not state["cache"]:
        # An interrupted cache is regenerated from scratch, not double-counted.
        if cache.exists():
            shutil.rmtree(cache)
        capture(model, tokenizer, rows, prefix, base_blocks, precision, cfg, cache)
        state["cache"] = {
            str(p.relative_to(output)): file_hash(p) for p in cache.rglob("*") if p.is_file()
        }
        save_json(state_path, state)
    research = output / "research-hf"
    research.mkdir(exist_ok=True)
    packed = output / "packed"
    packed.mkdir(exist_ok=True)
    student = output / "student-inputs"
    student.mkdir(exist_ok=True)
    devices = cfg.get("window_devices", [cfg.get("device", "cuda:0")])
    steps = int(cfg.get("steps", 100))
    width = int(cfg.get("window_size", 2 if cfg["preset"] == "extreme" else 1))
    stride = int(cfg.get("window_stride", 1))
    checkpoints = output / "training-checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for start, end, commit in windows(len(base_blocks), min(width, len(base_blocks)), stride):
        if start < state["next_block"]:
            continue
        blocks = [
            copy.deepcopy(base_blocks[i]).to(devices[(i - start) % len(devices)])
            for i in range(start, end)
        ]
        quantizers = {}
        hessians = {}
        for offset, block in enumerate(blocks):
            index = start + offset
            path = cache / str(index) / "hessians.safetensors"
            hessians.update(load_file(str(path)) if path.exists() else {})
            for name, weight in block.named_parameters():
                full = f"{prefix}.{index}.{name}"
                if full not in precision["tensors"]:
                    continue
                quantizers[full] = ProjectionQuantizer(
                    weight, precision["tensors"][full], components, hessians.get(full), cfg
                )
        if not quantizers:
            raise ValueError(f"No quantizers in window {start}:{end}")
        optimizer = Lion(
            [p for q in quantizers.values() for p in q.parameters() if p.requires_grad],
            lr=float(cfg.get("masks_lr", 0.0002)),
        )
        # Restore overlap factors and exact active-window optimiser/RNG state.
        first_step = 0
        active = output / state["active"]["file"] if state.get("active") else None
        if active is not None:
            old = torch.load(active, map_location="cpu", weights_only=True)
            if old["start"] == start:
                if old["identity"] != plan["identity"]:
                    raise ValueError("Active window identity mismatch")
                for name, values in old["quantizers"].items():
                    quantizers[name].load_state_dict(values)
                optimizer.load_state_dict(old["optimizer"])
                torch.set_rng_state(old["rng"])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(old["cuda_rng"])
                first_step = old["step"]
            else:
                for name in quantizers.keys() & old["quantizers"].keys():
                    quantizers[name].load_state_dict(old["quantizers"][name])
        metrics = []
        for step in range(first_step, steps):
            optimizer.zero_grad()
            schedule(quantizers, step, steps)
            for group in optimizer.param_groups:
                group["lr"] = float(cfg.get("masks_lr", 0.0002)) * (
                    0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2
                )
            sample = step % len(rows)
            initial = (
                torch.load(student / f"block-{start}-sample-{sample}.pt", weights_only=True)
                if start
                else torch.load(cache / "0" / f"sample-{sample}.pt", weights_only=True)
            )
            if start == 0:
                initial = (
                    initial["args"][0] if initial["args"] else initial["kwargs"]["hidden_states"]
                )
            result = run_blocks(blocks, quantizers, prefix, start, initial, cache, sample, devices)
            target = torch.load(cache / str(end - 1) / f"target-{sample}.pt", weights_only=True).to(
                result.device
            )
            loss = torch.nn.functional.mse_loss(result.float(), target.float())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite window reconstruction loss")
            loss.backward()
            optimizer.step()
            metrics.append(float(loss.detach()))
            print(
                json.dumps({"window": [start, end], "step": step + 1, "loss": metrics[-1]}),
                flush=True,
            )
            if (step + 1) % int(cfg.get("checkpoint_steps", 25)) == 0 or step + 1 == steps:
                checkpoint = checkpoints / f"window-{start}-step-{step + 1}.pt"
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(
                    {
                        "identity": plan["identity"],
                        "start": start,
                        "step": step + 1,
                        "quantizers": {n: q.state_dict() for n, q in quantizers.items()},
                        "optimizer": optimizer.state_dict(),
                        "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else [],
                    },
                    temporary,
                )
                temporary.replace(checkpoint)
                state["active"] = {
                    "file": str(checkpoint.relative_to(output)),
                    "sha256": file_hash(checkpoint),
                }
                save_json(state_path, state)
                if active is not None and active != checkpoint:
                    active.unlink(missing_ok=True)
                active = checkpoint
                if (
                    cfg.get("proof_mode")
                    and start == 0
                    and step + 1 == cfg.get("proof_interrupt_step")
                    and first_step == 0
                ):
                    raise RuntimeError("PROOF_CHECKPOINT_INTERRUPTION")
        for q in quantizers.values():
            q.eval()
        # Commit only the advancing prefix; overlapping blocks train again.
        for index in range(start, commit):
            offset = index - start
            weights = blocks[offset].state_dict()
            payload = {f"{prefix}.{index}.{name}": value.cpu() for name, value in weights.items()}
            for name, q in quantizers.items():
                if not name.startswith(f"{prefix}.{index}."):
                    continue
                payload[name] = q().detach().cpu()
                filename = packed / (digest(name) + ".safetensors")
                save_file(q.packed(), str(filename))
                state["completed"][name] = {
                    "file": str(filename.relative_to(output)),
                    "sha256": file_hash(filename),
                    "spec": q.spec,
                    "shape": list(q.shape),
                }
            save_file(
                payload, str(research / f"layer-{index:04d}.safetensors"), metadata={"format": "pt"}
            )
            path = research / f"layer-{index:04d}.safetensors"
            state.setdefault("exports", {})[str(path.relative_to(output))] = file_hash(path)
        with torch.no_grad():
            for sample in range(len(rows)):
                initial = (
                    torch.load(student / f"block-{start}-sample-{sample}.pt", weights_only=True)
                    if start
                    else torch.load(cache / "0" / f"sample-{sample}.pt", weights_only=True)
                )
                if not start:
                    initial = (
                        initial["args"][0]
                        if initial["args"]
                        else initial["kwargs"]["hidden_states"]
                    )
                result = run_blocks(
                    blocks[: commit - start],
                    quantizers,
                    prefix,
                    start,
                    initial,
                    cache,
                    sample,
                    devices,
                )
                torch.save(result.cpu(), student / f"block-{commit}-sample-{sample}.pt")
                path = student / f"block-{commit}-sample-{sample}.pt"
                state.setdefault("exports", {})[str(path.relative_to(output))] = file_hash(path)
        state["next_block"] = commit
        save_json(state_path, state)
        del blocks, quantizers, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Non-block weights remain unmodified. No model/dataset is committed to git.
    outer = {
        n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith(prefix + ".")
    }
    save_file(outer, str(research / "outer.safetensors"), metadata={"format": "pt"})
    for p in model_path.iterdir():
        if (
            p.is_file()
            and p.suffix in {".json", ".txt", ".model", ".jinja"}
            and not p.name.endswith(".safetensors.index.json")
        ):
            shutil.copy2(p, research / p.name)
    from appliance.quant.mixed_job import sources

    index = {n: p.name for n, p in sources(research).items()}
    save_json(research / "model.safetensors.index.json", {"metadata": {}, "weight_map": index})
    manifest = {
        "status": "succeeded",
        "implementation": "pinned_gsq_catq_sliding_adapter",
        "upstream": components[3],
        "identity": plan["identity"],
        "precision_map": str(output / "precision-map.json"),
        "research_model": str(research),
        "runtime_ready": False,
        "upstream_scaleq_reproduction": False,
        "tensors": state["completed"],
        "native_context_target": 262144,
        "synthetic_proof": bool(cfg.get("proof_mode")),
        "research_files": {
            p.name: file_hash(p) for p in research.iterdir() if p.is_file()
        },
    }
    save_json(output / "mixed-manifest.json", manifest)
    state["status"] = "succeeded"
    save_json(state_path, state)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(json.dumps(run(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
