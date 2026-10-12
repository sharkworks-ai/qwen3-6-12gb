"""Generate and checkpoint source-model reasoning traces for calibration."""

from __future__ import annotations

import argparse
import json
import os

from appliance.gpu import compute_dtype
from appliance.quant.mixed_job import file_hash
from appliance.quant.precision import digest
from appliance.stages.common import data_path, load_config, save_json


def run(cfg: dict):
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    source = data_path(cfg["source_model"], root)
    prompts = data_path(cfg["prompts_file"], root)
    output = data_path(cfg["output_file"], root)
    if output == prompts:
        raise ValueError("Trace output must differ from prompts")
    identity = digest(
        {
            "config": {k: v for k, v in cfg.items() if k != "resume"},
            "prompts": file_hash(prompts),
            "model": {p.name: file_hash(p) for p in source.glob("*.safetensors")},
            "model_config": file_hash(source / "config.json"),
        }
    )
    state_path = output.with_suffix(".progress.json")
    # "rows" counts prompts processed and "written" traces kept; they differ once an
    # incomplete trace is skipped.
    state = {"identity": identity, "rows": 0, "written": 0, "skipped": []}
    if state_path.exists():
        if not cfg.get("resume"):
            raise ValueError("Trace output exists. Enable Resume.")
        state = load_config(state_path)
        if state["identity"] != identity:
            raise ValueError("Trace resume config or source mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.read_text().splitlines() if output.exists() else []
    # Drop a row written after the last atomic progress checkpoint.
    written = state.get("written", state["rows"])
    output.write_text("".join(line + "\n" for line in existing[:written]))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=compute_dtype(cfg.get("device", "cuda:0")),
        device_map="auto",
        trust_remote_code=False,
        max_memory={int(k) if str(k).isdigit() else k: v for k, v in cfg["max_memory"].items()}
        if cfg.get("max_memory")
        else None,
    ).eval()
    rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    rows = rows[: int(cfg.get("max_samples", 128))]
    if not rows:
        raise ValueError("No calibration prompts")
    with output.open("a") as stream, torch.no_grad():
        for index in range(state["rows"], len(rows)):
            messages = rows[index].get("messages") or [
                {"role": "user", "content": rows[index]["prompt"]}
            ]
            if any(m["role"] == "assistant" for m in messages):
                raise ValueError("AYOT input must be source prompts, without teacher answers")
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            tokens = tokenizer(text, return_tensors="pt").to(
                model.get_input_embeddings().weight.device
            )
            if tokens.input_ids.shape[-1] > int(cfg.get("max_prompt_tokens", 2048)):
                raise ValueError("AYOT prompt exceeds max_prompt_tokens")
            result = model.generate(
                **tokens,
                # Coding prompts with several requirements often reason past 2048 tokens.
                max_new_tokens=int(cfg.get("max_new_tokens", 4096)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                result[0, tokens.input_ids.shape[-1] :], skip_special_tokens=False
            )
            if "</think>" in text + answer:
                stream.write(
                    json.dumps({"text": text + answer, "teacher_model": str(source)}) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
                written += 1
            else:
                # One long-winded prompt should not fail the stage; unfinished reasoning is
                # not a usable trace, so leave it out and record it.
                state.setdefault("skipped", []).append(index)
            state["rows"] = index + 1
            state["written"] = written
            save_json(state_path, state)
    complete = float(cfg.get("min_complete_fraction", 0.75))
    if written < max(1, complete * len(rows)):
        raise ValueError(
            f"Only {written} of {len(rows)} teacher traces completed reasoning within "
            f"{int(cfg.get('max_new_tokens', 4096))} tokens. Increase max_new_tokens."
        )
    manifest = {
        "kind": "ayot",
        "teacher_model": str(source),
        "rows": written,
        "prompts": state["rows"],
        "skipped_incomplete": state.get("skipped", []),
        "identity": identity,
        "sha256": file_hash(output),
    }
    save_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    print(json.dumps(run(load_config(parser.parse_args().config)), indent=2))


if __name__ == "__main__":
    main()
