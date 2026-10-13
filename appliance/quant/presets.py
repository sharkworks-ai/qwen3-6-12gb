from __future__ import annotations


def preset(name: str) -> dict:
    if name not in {"aggressive", "extreme"}:
        raise ValueError(name)
    return {
        "preset": name,
        "engine": "upstream_window",
        "window_size": 2 if name == "extreme" else 1,
        "window_stride": 1,
        "window_devices": ["cuda:0", "cuda:1"],
        "checkpoint_steps": 25,
        "masks_lr": 0.0002,
        "gptq_damp": 0.1,
        "gptq_block_size": 128,
        "catq_rank": 4,
        "catq_alpha": 4,
        "source_model": "/data/checkpoints/qwen36-merged",
        "output_dir": f"/data/artifacts/{name}-upstream",
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
        # 2-bit GSQ, not 3: a 3-bit quantizer keeps state for 8 levels per weight, and
        # layers 0 and 1 share one 2-block window, which then exceeds 2x32 GB GPUs.
        "sensitive_tensors": [{"pattern": r"\.layers\.(0|1)\..*\.experts\.", "bits": 2}]
        if name == "extreme"
        else [],
        "recovery": {
            "enabled": name == "extreme",
            "dataset": "/data/datasets/recovery.jsonl",
            # LoRA QAT shards one process across the GPUs.
            "num_processes": 1,
            "trainable": "lora",
            "max_steps": 200,
            "save_steps": 50,
            # 8192 tokens ran out of memory on 2x RTX 5090 (about 30 GB per GPU);
            # 4096 peaks at 29 GB and truncates 3% of the r0b0tlab recovery rows.
            "max_length": 4096,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.000005,
        },
    }
