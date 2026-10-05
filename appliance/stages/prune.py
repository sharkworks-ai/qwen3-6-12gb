from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch

from appliance.stages.common import load_config, save_json

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.(?:experts\.(?:gate_up_proj|down_proj)|gate\.weight)$")


def build_keep_map(profile: dict, keep_experts: int, mass_weight: float) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for layer_str, stats in profile["layers"].items():
        counts = torch.tensor(stats["count"], dtype=torch.float64)
        mass = torch.tensor(stats["routing_mass"], dtype=torch.float64)
        if keep_experts > counts.numel():
            raise ValueError(f"keep_experts={keep_experts} exceeds layer expert count={counts.numel()}")
        count_score = counts / max(float(counts.sum()), 1.0)
        mass_score = mass / max(float(mass.sum()), 1e-12)
        score = (1.0 - mass_weight) * count_score + mass_weight * mass_score
        keep = torch.topk(score, k=keep_experts, largest=True, sorted=False).indices
        result[int(layer_str)] = sorted(int(x) for x in keep.tolist())
    return result


def ensure_snapshot(source: str, token: str | None) -> Path:
    candidate = Path(source)
    if candidate.exists():
        return candidate.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=source, token=token)).resolve()


def copy_non_weight_files(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.name in {"model.safetensors", "model.safetensors.index.json", "config.json"}:
            continue
        target = output / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    profile = json.loads(Path(config["profile"]).read_text(encoding="utf-8"))
    keep_experts = int(config.get("keep_experts", 128))
    mass_weight = float(config.get("routing_mass_weight", 0.5))
    keep_map = build_keep_map(profile, keep_experts, mass_weight)

    output = Path(config.get("output_dir", "/data/checkpoints/qwen36-pruned"))
    if config.get("dry_run"):
        save_json(output.parent / "prune_plan.json", {
            "keep_experts": keep_experts,
            "layers": {str(k): v for k, v in keep_map.items()},
        })
        print(json.dumps({"dry_run": True, "layers": len(keep_map), "keep_experts": keep_experts}))
        return

    source = ensure_snapshot(config.get("model", "Qwen/Qwen3.6-35B-A3B"), os.environ.get("HF_TOKEN") or None)
    output.mkdir(parents=True, exist_ok=True)
    copy_non_weight_files(source, output)

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    elif (source / "model.safetensors").exists():
        index = None
        shard_names = ["model.safetensors"]
    else:
        raise FileNotFoundError("No safetensors checkpoint found")

    from safetensors.torch import load_file, save_file

    total_size = 0
    for shard_name in shard_names:
        tensors = load_file(str(source / shard_name), device="cpu")
        rewritten = {}
        for name, tensor in tensors.items():
            match = LAYER_RE.search(name)
            if match:
                layer = int(match.group(1))
                keep = keep_map.get(layer)
                if keep is not None:
                    index_tensor = torch.tensor(keep, dtype=torch.long)
                    # Expert banks and router weights all use expert dimension 0.
                    tensor = tensor.index_select(0, index_tensor)
            rewritten[name] = tensor.contiguous()
            total_size += tensor.numel() * tensor.element_size()
        save_file(rewritten, str(output / shard_name), metadata={"format": "pt"})
        del tensors, rewritten

    if index is not None:
        index.setdefault("metadata", {})["total_size"] = total_size
        save_json(output / "model.safetensors.index.json", index)

    model_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    target = model_config.get("text_config", model_config)
    target["num_experts"] = keep_experts
    if int(target.get("num_experts_per_tok", 8)) > keep_experts:
        raise ValueError("num_experts_per_tok cannot exceed retained expert count")
    save_json(output / "config.json", model_config)
    save_json(output / "expert_pruning_map.json", {
        "source": str(source),
        "profile": config["profile"],
        "keep_experts": keep_experts,
        "routing_mass_weight": mass_weight,
        "layers": {str(k): v for k, v in keep_map.items()},
    })
    print(json.dumps({"output": str(output), "keep_experts": keep_experts, "bytes": total_size}))


if __name__ == "__main__":
    main()
