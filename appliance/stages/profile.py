from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from appliance.stages.common import load_config, resolve_model_source, save_json
from appliance.stages.dataset_io import load_training_dataset, maybe_limit


def _conversation_text(processor, row: dict) -> str:
    messages = row.get("messages")
    if messages is not None:
        return processor.apply_chat_template(
            messages,
            tools=row.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )
    for key in ("text", "prompt", "content"):
        if key in row and row[key]:
            return str(row[key])
    raise ValueError("Dataset rows must contain messages, text, prompt, or content")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    if config.get("dry_run"):
        print(json.dumps({"dry_run": True, "stage": "profile", "config": config}, indent=2))
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_source = resolve_model_source(config.get("model", "Qwen/Qwen3.6-35B-A3B"))
    output = Path(config.get("output", "/data/artifacts/expert_profile.json"))
    output.parent.mkdir(parents=True, exist_ok=True)

    quant = None
    if config.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        token=os.environ.get("HF_TOKEN") or None,
    )
    if config.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config["adapter"], is_trainable=False)
    processor = AutoProcessor.from_pretrained(model_source, token=os.environ.get("HF_TOKEN") or None)
    dataset = maybe_limit(load_training_dataset(config), config)

    counts: dict[int, torch.Tensor] = {}
    masses: dict[int, torch.Tensor] = {}
    handles = []

    for name, module in model.named_modules():
        if not name.endswith(".mlp.gate"):
            continue
        if not hasattr(module, "num_experts"):
            continue
        layer_match = None
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_match = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if layer_match is None:
            continue
        num_experts = int(module.num_experts)
        counts[layer_match] = torch.zeros(num_experts, dtype=torch.long)
        masses[layer_match] = torch.zeros(num_experts, dtype=torch.float64)

        def hook(_module, _inputs, output, layer=layer_match):
            # Qwen3.6 router returns (logits, normalized top-k weights, selected experts).
            _logits, weights, selected = output
            selected_cpu = selected.detach().to("cpu").reshape(-1)
            weights_cpu = weights.detach().to("cpu", dtype=torch.float64).reshape(-1)
            counts[layer] += torch.bincount(selected_cpu, minlength=counts[layer].numel())
            masses[layer] += torch.bincount(
                selected_cpu,
                weights=weights_cpu,
                minlength=masses[layer].numel(),
            )

        handles.append(module.register_forward_hook(hook))

    max_tokens = int(config.get("max_length", 8192))
    max_samples = int(config.get("max_samples", 512))
    processed = 0
    with torch.no_grad():
        for row in dataset:
            if processed >= max_samples:
                break
            text = _conversation_text(processor, row)
            inputs = processor(
                text=[text],
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            # device_map may shard the model; input embeddings establish the input device.
            device = model.get_input_embeddings().weight.device
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            model(**inputs, use_cache=False)
            processed += 1

    for handle in handles:
        handle.remove()

    profile = {
        "model": model_source,
        "samples": processed,
        "layers": {
            str(layer): {
                "count": counts[layer].tolist(),
                "routing_mass": masses[layer].tolist(),
            }
            for layer in sorted(counts)
        },
    }
    save_json(output, profile)
    print(json.dumps({"output": str(output), "samples": processed, "layers": len(counts)}))


if __name__ == "__main__":
    main()
