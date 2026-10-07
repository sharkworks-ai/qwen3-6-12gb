"""Capture actual HF block calls, teacher targets and full projection Hessians."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils._python_dispatch import TorchDispatchMode


def move(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported block-call object: {type(value).__name__}")


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


def find_blocks(model):
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and name.endswith(".layers")
    ]
    # Exclude vision encoder lists in multimodal checkpoints.
    candidates = [(n, m) for n, m in candidates if "visual" not in n and "vision" not in n]
    if len(candidates) != 1:
        raise ValueError(f"Unable to uniquely find decoder blocks: {[n for n, _ in candidates]}")
    return candidates[0]


class FinishedCapture(Exception):
    pass


class HessianCapture(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.weights = {}
        self.sums = {}
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        operands = args[1:3] if func == torch.ops.aten.addmm.default else args[:2]
        if func in {
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        }:
            for index, operand in enumerate(operands):
                key = operand.untyped_storage().data_ptr()
                if key in self.weights:
                    name, columns = self.weights[key]
                    inputs = operands[1 - index]
                    if index == 0:
                        inputs = inputs.transpose(-1, -2)
                    if inputs.shape[-1] == columns:
                        inputs = inputs.float().reshape(-1, columns)
                        gram = (inputs.T @ inputs).cpu()
                        self.sums[name] = self.sums.get(name, torch.zeros_like(gram)) + gram
                        self.counts[name] = self.counts.get(name, 0) + len(inputs)
        return func(*args, **(kwargs or {}))


def capture(model, tokenizer, rows, prefix, blocks, precision, cfg, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda:0")
    projection_names = set(precision["tensors"])
    mode = HessianCapture()
    handles = []
    current_sample = 0
    for index, block in enumerate(blocks):
        folder = directory / str(index)
        folder.mkdir(exist_ok=True)

        def pre(module, args, kwargs, index=index, folder=folder):
            torch.save(
                {"args": move(args, "cpu"), "kwargs": move(kwargs, "cpu")},
                folder / f"sample-{current_sample}.pt",
            )
            module.to(device)
            mode.weights.clear()
            mode.sums.clear()
            mode.counts.clear()
            for name, parameter in module.named_parameters():
                full = f"{prefix}.{index}.{name}"
                if full in projection_names and precision["tensors"][full]["bits"] != 1.58:
                    mode.weights[parameter.untyped_storage().data_ptr()] = (
                        full,
                        parameter.shape[-1],
                    )
            return move(args, device), move(kwargs, device)

        def post(module, args, output, index=index, folder=folder):
            torch.save(hidden(output).detach().cpu(), folder / f"target-{current_sample}.pt")
            hessian_path = folder / "hessians.safetensors"
            old = load_file(str(hessian_path)) if hessian_path.exists() else {}
            counts_path = folder / "counts.json"
            counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
            for name, gram in mode.sums.items():
                old[name] = old.get(name, torch.zeros_like(gram)) + gram
                counts[name] = counts.get(name, 0) + mode.counts[name]
            if old:
                save_file(old, str(hessian_path))
                counts_path.write_text(json.dumps(counts))
            mode.weights.clear()
            module.to("cpu")
            if index == len(blocks) - 1:
                raise FinishedCapture()
            return move(output, "cpu")

        handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(block.register_forward_hook(post))
        if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
            experts = block.mlp.experts
            if hasattr(experts, "config"):
                experts.config._experts_implementation = "eager"
    try:
        with torch.no_grad(), mode:
            for current_sample, row in enumerate(rows):
                text = row.get("text") or tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg.get("calibration_length", 2048)),
                )
                try:
                    model(**tokens, use_cache=False)
                except FinishedCapture:
                    pass
    finally:
        for handle in handles:
            handle.remove()
        for block in blocks:
            block.to("cpu")
    for index, block in enumerate(blocks):
        folder = directory / str(index)
        needed = {
            n
            for n, s in precision["tensors"].items()
            if n.startswith(f"{prefix}.{index}.") and s["bits"] != 1.58
        }
        path = folder / "hessians.safetensors"
        data = load_file(str(path)) if path.exists() else {}
        if needed - set(data):
            raise ValueError(f"Unobserved GPTQ projections: {sorted(needed - set(data))[:8]}")
        counts = json.loads((folder / "counts.json").read_text()) if data else {}
        if data:
            save_file({n: 2 * value / counts[n] for n, value in data.items()}, str(path))
