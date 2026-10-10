from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(target)


def data_path(value: str, data_root: str = "/data") -> Path:
    root = Path(data_root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must stay under {root}: {path}")
    return path


def device_memory(max_memory: dict | None, *, gpu_only: bool = False) -> dict | None:
    """Accelerate max_memory from JSON, where GPU indices arrive as string keys."""
    if not max_memory:
        return None
    return {
        int(k) if str(k).isdigit() else k: v
        for k, v in max_memory.items()
        if not (gpu_only and not str(k).isdigit())
    }


def keep_head_with_embeddings(model) -> None:
    """Move a device_map-split model's LM head onto the embeddings' device.

    Trainer requires the loss on the first device, and TRL's fused loss runs a Triton
    kernel on the current device; device_map="auto" puts the head on the last GPU.
    """
    head = model.get_output_embeddings()
    first = model.get_input_embeddings().weight.device
    if head.weight.device != first:
        head.to(first)
        hook = getattr(head, "_hf_hook", None)
        if hook is not None:
            hook.execution_device = first


def resolve_model_source(model: str, data_root: str = "/data") -> str:
    if model.startswith("/") or model.startswith("."):
        return str(data_path(model, data_root))
    return model
