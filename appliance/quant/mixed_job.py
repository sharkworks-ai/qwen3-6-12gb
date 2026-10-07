"""Resumable mixed precision reconstruction and explicit BF16 research export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from appliance.quant.precision import build_map, digest, validate
from appliance.stages.common import data_path, load_config, save_json


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sources(model: Path) -> dict[str, Path]:
    from safetensors import safe_open

    result = {}
    for path in sorted(model.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                if name in result:
                    raise ValueError(f"Duplicate checkpoint tensor {name}")
                result[name] = path
    if not result:
        raise ValueError("A local, unquantized HF safetensors checkpoint is required")
    return result


def prepare(cfg: dict) -> tuple[Path, Path, dict, dict]:
    validate(cfg)
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    model = data_path(cfg["source_model"], root)
    output = data_path(cfg["output_dir"], root)
    if output == model or model in output.parents or output in model.parents:
        raise ValueError("Source and output must be separate directories")
    files = sources(model)
    config = json.loads((model / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("Input is already quantized; use the merged BF16 checkpoint")
    precision = build_map(list(files), cfg)
    calibration = data_path(cfg["calibration_file"], root)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    if cfg["preset"] == "extreme":
        metadata = json.loads(calibration.with_suffix(".manifest.json").read_text())
        synthetic = cfg.get("proof_mode") and config.get("qwen12g_proof_model") is True
        if synthetic:
            if metadata.get("kind") != "synthetic_proof_fixture":
                raise ValueError("Proof mode requires explicitly synthetic fixtures")
        elif metadata.get("kind") != "ayot" or metadata.get("teacher_model") != str(
            cfg.get("teacher_model", model)
        ):
            raise ValueError("Extreme requires AYOT traces from the source checkpoint")
        if metadata.get("sha256") != file_hash(calibration):
            raise ValueError("AYOT calibration hash mismatch")
    identity = {
        "config": {k: v for k, v in cfg.items() if k not in {"resume", "dry_run"}},
        "source": {str(p.name): file_hash(p) for p in sorted(set(files.values()))},
        "model_config": file_hash(model / "config.json"),
        "calibration": file_hash(calibration),
        "precision": precision,
    }
    return model, output, files, {"identity": digest(identity), **identity}


def run(cfg: dict) -> dict:
    from appliance.quant.locking import exclusive

    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    output = data_path(cfg["output_dir"], root)
    model = data_path(cfg["source_model"], root)
    if output == model or model in output.parents or output in model.parents:
        raise ValueError("Source and output must be separate directories")
    with exclusive(output / ".reconstruction.lock"):
        return _run(cfg)


def _run(cfg: dict) -> dict:
    model, output, files, plan = prepare(cfg)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "plan.json", plan)
    if cfg.get("dry_run"):
        return {"status": "planned", "plan": str(output / "plan.json")}
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from appliance.quant.lowbit import dequantize, pack_codes, reconstruct
    from appliance.quant.moments import collect_moments

    progress_path = output / "progress.json"
    state = {"identity": plan["identity"], "completed": {}}
    if progress_path.exists():
        if not cfg.get("resume"):
            raise ValueError("Output contains a run. Enable Resume or choose a new directory.")
        state = load_config(progress_path)
        if state["identity"] != plan["identity"]:
            raise ValueError("Resume rejected: source, calibration or config changed")
    save_json(output / "precision-map.json", plan["precision"])
    # Calibration is mandatory, including for expert tensors; never silently RTN.
    moments_path = output / "moments.safetensors"
    if not moments_path.exists():
        calibration = data_path(
            cfg["calibration_file"], os.environ.get("QWEN12G_DATA_ROOT", "/data")
        )
        collect_moments(model, calibration, plan["precision"], cfg, moments_path)
    moments = load_file(str(moments_path))
    moments_hash = file_hash(moments_path)
    if state.get("moments_sha256") and state["moments_sha256"] != moments_hash:
        raise ValueError("Resume calibration moments failed integrity check")
    state["moments_sha256"] = moments_hash
    save_json(progress_path, state)
    artifacts = output / "packed"
    artifacts.mkdir(exist_ok=True)
    for name, spec in plan["precision"]["tensors"].items():
        artifact = artifacts / (hashlib.sha256(name.encode()).hexdigest() + ".safetensors")
        previous = state["completed"].get(name)
        if previous:
            if not artifact.exists() or file_hash(artifact) != previous["sha256"]:
                raise ValueError(f"Resume artifact failed integrity check: {name}")
            continue
        with safe_open(files[name], framework="pt", device="cpu") as f:
            weight = f.get_tensor(name)
        if name not in moments:
            raise ValueError(f"Calibration never reached tensor {name}; add representative traces")
        shape, codes_parts, scale_parts, errors = list(weight.shape), [], [], []
        rows = weight.reshape(-1, weight.shape[-1])
        # Limit logit memory independently of the total expert-bank size.
        for offset in range(0, len(rows), int(cfg.get("row_chunk", 64))):
            chunk = rows[offset : offset + int(cfg.get("row_chunk", 64))]
            codes, scales, error, _ = reconstruct(
                chunk.to(cfg.get("device", "cuda:0")),
                spec,
                steps=int(cfg.get("steps", 100)),
                lr=float(cfg.get("learning_rate", 0.01)),
                seed=int(cfg.get("seed", 42)),
                importance=moments[name],
            )
            codes_parts.append(codes)
            scale_parts.append(scales)
            errors.append(error)
        packed = pack_codes(torch.cat(codes_parts), spec["bits"])
        scales = torch.cat(scale_parts)
        temp = artifact.with_suffix(".tmp")
        save_file({"codes": packed, "scales": scales}, str(temp))
        temp.replace(artifact)
        state["completed"][name] = {
            "file": str(artifact.relative_to(output)),
            "sha256": file_hash(artifact),
            "shape": shape,
            "spec": spec,
            "weighted_mse": sum(errors) / len(errors),
            "stored_bytes": artifact.stat().st_size,
        }
        save_json(progress_path, state)
        print(json.dumps({"tensor": name, **state["completed"][name]}), flush=True)
    # Dequantized HF checkpoint is for validation/QAT only, never labelled 12GB.
    recovered = output / "research-hf"
    recovered.mkdir(exist_ok=True)
    index = {}
    total = 0
    for i, (name, source) in enumerate(files.items()):
        if name in state["completed"]:
            item = state["completed"][name]
            payload = load_file(str(output / item["file"]))
            tensor = dequantize(
                payload["codes"], payload["scales"], item["spec"], item["shape"]
            ).to(torch.bfloat16)
        else:
            with safe_open(source, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
        shard = f"model-{i:06d}.safetensors"
        save_file({name: tensor.contiguous()}, str(recovered / shard), metadata={"format": "pt"})
        index[name] = shard
        total += tensor.numel() * tensor.element_size()
    for source in model.iterdir():
        if (
            source.is_file()
            and source.suffix in {".json", ".txt", ".model", ".jinja"}
            and not source.name.endswith(".safetensors.index.json")
        ):
            shutil.copy2(source, recovered / source.name)
    save_json(
        recovered / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": index},
    )
    manifest = {
        "status": "succeeded",
        "implementation": "experimental_local_gsq_ayot_reference",
        "upstream_scaleq_reproduction": False,
        "runtime_ready": False,
        "native_context_target": 262144,
        "identity": plan["identity"],
        "precision_map": str(output / "precision-map.json"),
        "research_model": str(recovered),
        "packed_bytes": sum(x["stored_bytes"] for x in state["completed"].values()),
        "notes": [
            "Packed shards have no production inference kernel.",
            "BF16 research export is for evaluation and QAT, not memory savings.",
            "No coding, tool-loop or 262K release gate has been passed.",
        ],
        "tensors": state["completed"],
    }
    save_json(output / "mixed-manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
