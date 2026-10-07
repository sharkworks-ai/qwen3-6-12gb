"""Small single-device bootstrap/QAT loop with optimizer and RNG resume."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from appliance.qat.parametrize import apply_precision_map, remove_fake_quant


def train(cfg, source, output, *, precision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = cfg["device"]
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    torch.manual_seed(int(cfg["seed"]))
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=dtype, attn_implementation="eager", trust_remote_code=False
    ).to(device)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if precision is not None:
        apply_precision_map(model, json.loads(Path(precision).read_text()))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    rows = [
        json.loads(line)["text"]
        for line in Path(cfg["training_file"]).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Empty training data")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    first = 0
    losses = []
    if checkpoint.exists():
        if not cfg.get("resume"):
            raise ValueError("Training exists; use Resume")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        if device != "cpu":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        first, losses = saved["step"], saved["losses"]
    for step in range(first, int(cfg["training_steps"])):
        tokens = tokenizer(
            rows[step % len(rows)],
            return_tensors="pt",
            truncation=True,
            max_length=int(cfg["sequence_length"]),
        ).to(device)
        labels = tokens.input_ids.clone()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**tokens, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(json.dumps({"training_step": step + 1, "loss": losses[-1]}), flush=True)
        if (step + 1) % int(cfg["checkpoint_steps"]) == 0 or step + 1 == int(cfg["training_steps"]):
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "losses": losses,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device != "cpu"
                    else torch.empty(0, dtype=torch.uint8),
                },
                temporary,
            )
            temporary.replace(checkpoint)
    if precision is not None:
        remove_fake_quant(model)
    model.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    return {
        "steps": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "qat": precision is not None,
    }
