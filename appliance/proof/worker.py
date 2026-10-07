"""Isolated proof stages. Fixed operations only, no user-provided commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from appliance.stages.common import load_config, save_json


def execute(cfg):
    operation = cfg["operation"]
    stage = Path(cfg["stage_dir"])
    if cfg["device"] == "cpu":
        if not cfg.get("cpu_test"):
            raise ValueError("Proof jobs require a CUDA GPU")
        torch.set_num_threads(1)
        # CPU CI only: GSQ assumes CUDA RNG replay, arithmetic stays unchanged.
        original_fork = torch.random.fork_rng
        torch.cuda.get_rng_state = lambda **k: torch.get_rng_state()
        torch.cuda.set_rng_state = lambda state, **k: torch.set_rng_state(state)
        torch.random.fork_rng = lambda **k: original_fork(devices=[])
    elif not torch.cuda.is_available():
        raise ValueError("CUDA GPU unavailable")
    if operation == "generate":
        from appliance.proof.model import generate

        result = generate(cfg, stage)
    elif operation == "train":
        from appliance.proof.training import train

        result = train(cfg, cfg["source_model"], stage, precision=cfg.get("precision_map"))
    elif operation == "reconstruct":
        from appliance.quant.window_job import run

        try:
            result = run(cfg["quant_config"])
        except RuntimeError as error:
            if str(error) == "PROOF_CHECKPOINT_INTERRUPTION":
                raise SystemExit(75) from error
            raise
    elif operation == "evaluate":
        from appliance.proof.evaluate import evaluate

        result = evaluate(
            cfg, cfg["source_model"], bundle=cfg.get("bundle"), baseline=cfg.get("baseline")
        )
        logits = result.pop("logits")
        torch.save({**result, "logits": logits}, stage / "logits.pt")
    elif operation == "pipeline":
        from appliance.quant.pipeline import run

        result = run(cfg["quant_config"])
    elif operation == "context":
        from appliance.proof.validation import context_check

        result = context_check(cfg)
    else:
        raise ValueError(f"Unknown proof operation {operation}")
    save_json(stage / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    result = execute(cfg)
    if cfg["operation"] in ("reconstruct", "pipeline"):
        result = {
            k: result[k] for k in ("status", "research_model", "runtime_ready") if k in result
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
