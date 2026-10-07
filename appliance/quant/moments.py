"""Bounded activation second moments for linear and fused MoE projections."""

from __future__ import annotations

import json
from pathlib import Path


def collect_moments(model_path: Path, calibration: Path, precision: dict, cfg: dict, output: Path):
    import torch
    from safetensors.torch import save_file
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=cfg.get("max_memory"),
        trust_remote_code=False,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    modules = dict(model.named_modules())
    # Calibration needs observable projection matmuls, not opaque grouped kernels.
    for module in modules.values():
        if (
            hasattr(module, "gate_up_proj")
            and hasattr(module, "down_proj")
            and hasattr(module, "config")
        ):
            module.config._experts_implementation = "eager"
    sums, counts, handles = {}, {}, []
    fused = {}

    def record(name, x):
        x = x.detach().float().reshape(-1, x.shape[-1])
        value = x.square().sum(0).cpu()
        sums[name] = sums.get(name, torch.zeros_like(value)) + value
        counts[name] = counts.get(name, 0) + len(x)

    for name in precision["tensors"]:
        module_name, parameter = name.rsplit(".", 1)
        module = modules[module_name]
        if parameter != "weight" or not isinstance(module, torch.nn.Linear):
            weight = getattr(module, parameter)
            if weight.ndim != 3:
                raise ValueError(f"Unsupported calibration operator: {name}")
            fused[weight.untyped_storage().data_ptr()] = (name, weight.shape[-1])
            continue

        def hook(mod, inputs, tensor_name=name):
            record(tensor_name, inputs[0])

        handles.append(module.register_forward_pre_hook(hook))

    class ExpertMoments(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # Eager mm/bmm uses views of fused bank tensors. Custom opaque
            # kernels are deliberately not guessed and fail coverage below.
            if func in {torch.ops.aten.mm.default, torch.ops.aten.bmm.default}:
                for i, operand in enumerate(args[:2]):
                    key = operand.untyped_storage().data_ptr()
                    if key in fused:
                        name, columns = fused[key]
                        activation = args[1 - i]
                        if i == 0:
                            activation = activation.transpose(-1, -2)
                        if activation.shape[-1] == columns:
                            record(name, activation)
            return func(*args, **(kwargs or {}))

    try:
        device = model.get_input_embeddings().weight.device
        with torch.no_grad(), ExpertMoments(), calibration.open() as stream:
            for i, line in enumerate(stream):
                if i >= int(cfg.get("max_samples", 128)):
                    break
                row = json.loads(line)
                text = row.get("text") or tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg.get("calibration_length", 2048)),
                ).to(device)
                model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    missing = set(precision["tensors"]) - set(sums)
    if missing:
        raise ValueError(f"Uncalibrated tensors ({len(missing)}): {sorted(missing)[:8]}")
    save_file({n: (sums[n] / counts[n]).clamp_min(1e-8) for n in sums}, str(output))
