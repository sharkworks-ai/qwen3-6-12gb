"""Held-out loss, packed parity and artifact/resource metrics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from appliance.runtime.packed import load_packed


@torch.inference_mode()
def evaluate(cfg, source, *, bundle=None, baseline=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if cfg["device"] == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if cfg["device"] != "cpu":
        torch.cuda.reset_peak_memory_stats(cfg["device"])
    if bundle:
        model = load_packed(bundle, device=cfg["device"], dtype=dtype, chunk_rows=cfg["chunk_rows"])
    else:
        options = {}
        if cfg.get("hf_device_map"):
            options["device_map"] = cfg["hf_device_map"]
            options["max_memory"] = {
                int(k) if str(k).isdigit() else k: v for k, v in cfg["max_memory"].items()
            }
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, attn_implementation="eager", trust_remote_code=False, **options
        ).eval()
        if not options:
            model = model.to(cfg["device"])
    input_device = model.get_input_embeddings().weight.device
    rows = [
        json.loads(line)["text"]
        for line in Path(cfg["heldout_file"]).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Empty held-out corpus")
    rows = rows[: int(cfg.get("eval_samples", len(rows)))]
    losses, outputs = [], []
    for text in rows:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(cfg.get("eval_length", cfg.get("sequence_length", 128))),
        ).to(input_device)
        result = model(**tokens, labels=tokens.input_ids, use_cache=False)
        if not torch.isfinite(result.loss) or not torch.isfinite(result.logits).all():
            raise ValueError("Non-finite evaluation")
        losses.append(float(result.loss))
        if cfg.get("capture_logits", True):
            outputs.append(result.logits.detach().float().cpu())
    payload = {
        "heldout_loss": sum(losses) / len(losses),
        "samples": len(rows),
        "runtime": "packed_eager" if bundle else "hf",
        "resident_tensor_bytes": sum(
            x.numel() * x.element_size() for x in list(model.parameters()) + list(model.buffers())
        ),
        "logits": outputs,
    }
    payload["inference_peak_allocated_mib"] = (
        torch.cuda.max_memory_allocated(cfg["device"]) / 1024**2 if cfg["device"] != "cpu" else None
    )
    if baseline is not None:
        reference = torch.load(baseline, weights_only=True)
        errors = [
            (a - b).square().mean() for a, b in zip(outputs, reference["logits"], strict=True)
        ]
        payload["baseline_logit_mse"] = float(torch.stack(errors).mean()) if errors else None
        payload["heldout_loss_delta"] = payload["heldout_loss"] - reference["heldout_loss"]
    if bundle and cfg.get("compare_export", True):
        # Reload the corresponding BF16 research export, sequentially so both
        # models are not GPU-resident during the memory measurement.
        del model
        if cfg["device"] != "cpu":
            torch.cuda.empty_cache()
        reference_model = (
            AutoModelForCausalLM.from_pretrained(
                source, dtype=dtype, attn_implementation="eager", trust_remote_code=False
            )
            .to(cfg["device"])
            .eval()
        )
        maximum = 0.0
        reference_argmax, packed_argmax = [], []
        for text, packed_logits in zip(rows, outputs, strict=True):
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(cfg.get("eval_length", cfg.get("sequence_length", 128))),
            ).to(cfg["device"])
            logits = reference_model(**tokens, use_cache=False).logits.float().cpu()
            maximum = max(maximum, float((logits - packed_logits).abs().max()))
            reference_argmax.append(logits.argmax(-1))
            packed_argmax.append(packed_logits.argmax(-1))
        payload["export_max_logit_error"] = maximum
        payload["export_argmax_agreement"] = sum(
            int((a == b).sum()) for a, b in zip(reference_argmax, packed_argmax)
        ) / sum(a.numel() for a in reference_argmax)
        # Scales are stored FP16, so the packed and BF16 research grid differ by
        # scale rounding. Report a dtype-dependent bound, not bitwise equivalence.
        payload["export_parity_pass"] = maximum <= (0.25 if dtype == torch.bfloat16 else 0.02)
    return payload
