"""Small hybrid MoE proof and full-model validation defaults."""

from __future__ import annotations


def defaults():
    return {
        "output_dir": "/data/artifacts/proof-12gb",
        "cuda_devices": "0",
        "device": "cuda:0",
        "seed": 42,
        "resume": False,
        "hidden_size": 512,
        "layers": 8,
        "experts": 16,
        "top_k": 2,
        "expert_width": 768,
        "sequence_length": 128,
        "samples": 8,
        "train_steps": 20,
        "reconstruct_steps": 10,
        "qat_steps": 10,
        "checkpoint_steps": 5,
        "learning_rate": 0.0003,
        "chunk_rows": 128,
        "vram_limit_gib": 11.5,
    }


def validate(cfg):
    if cfg["device"] != "cpu" and (
        cfg["device"] != "cuda:0" or not str(cfg["cuda_devices"]).isdigit()
    ):
        raise ValueError("Proof uses one numeric cuda_devices selection and logical device cuda:0")
    for name in (
        "train_steps",
        "reconstruct_steps",
        "qat_steps",
        "checkpoint_steps",
        "samples",
        "chunk_rows",
    ):
        if int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    hidden = int(cfg["hidden_size"])
    width = int(cfg["expert_width"])
    if hidden < 64 or hidden % 64 or width < 64 or width % 64:
        raise ValueError("Hidden and expert widths must be positive multiples of 64")
    if not 3 <= int(cfg["layers"]) <= 16 or not 2 <= int(cfg["experts"]) <= 32:
        raise ValueError("Proof model requires 3–16 layers and 2–32 experts")
    if not 1 <= int(cfg["top_k"]) <= int(cfg["experts"]):
        raise ValueError("Invalid top_k")
    if not 16 <= int(cfg["sequence_length"]) <= 2048:
        raise ValueError("Proof sequence_length must be 16–2048")
    if not 0 < float(cfg["vram_limit_gib"]) <= 12:
        raise ValueError("Proof VRAM limit must be between 0 and 12 GiB")
    # Bound configuration, rather than promising any GPU's actual fit.
    estimate = int(cfg["layers"]) * int(cfg["experts"]) * 3 * hidden * width
    if estimate > 300_000_000:
        raise ValueError("Proof expert parameters exceed the 300M safety bound")


def validation_defaults():
    return {
        "output_dir": "/data/artifacts/full-validation",
        "source_model": "/data/checkpoints/qwen36-merged",
        "calibration_file": "/data/datasets/calibration.jsonl",
        "ayot_file": "/data/datasets/ayot.jsonl",
        "recovery_dataset": "/data/datasets/recovery.jsonl",
        "heldout_file": "/data/datasets/heldout.jsonl",
        "cuda_devices": "0,1",
        "device": "cuda:0",
        "window_devices": ["cuda:0", "cuda:1"],
        "runtime_device": "cuda:0",
        "resume": False,
        "steps": 100,
        "max_samples": 128,
        "calibration_length": 2048,
        "eval_length": 512,
        "eval_samples": 8,
        "max_memory": {"0": "28GiB", "1": "28GiB", "cpu": "80GiB"},
        "context_lengths": [32768, 65536, 131072, 200000, 250000],
        "context_repeats": 5,
        "max_new_tokens": 8192,
        "recovery": {"num_processes": 2, "max_steps": 200, "save_steps": 50},
        "benchmark_results": "",
        "vram_limit_gib": 11.5,
    }
