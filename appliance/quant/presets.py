from __future__ import annotations


def preset(name: str) -> dict:
    if name not in {"aggressive", "extreme"}:
        raise ValueError(name)
    return {
        "preset": name,
        "source_model": "/data/checkpoints/qwen36-merged",
        "output_dir": f"/data/artifacts/{name}",
        "calibration_file": "/data/datasets/ayot.jsonl"
        if name == "extreme"
        else "/data/datasets/calibration.jsonl",
        "attention_bits": 4,
        "shared_bits": 4,
        "group_size": 64,
        "context_length": 262144,
        "steps": 100,
        "row_chunk": 64,
        "max_samples": 128,
        "calibration_length": 2048,
        "learning_rate": 0.01,
        "seed": 42,
        "device": "cuda:0",
        "cuda_devices": "0,1",
        "dry_run": True,
        "resume": False,
        "sensitive_tensors": [{"pattern": r"\.layers\.(0|1)\..*\.experts\.", "bits": 3}]
        if name == "extreme"
        else [],
        "recovery": {
            "enabled": name == "extreme",
            "dataset": "/data/datasets/recovery.jsonl",
            "num_processes": 2,
            "max_steps": 200,
            "save_steps": 50,
            "max_length": 8192,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.000005,
        },
    }
