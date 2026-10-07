"""Choose proof arithmetic that the selected device supports natively."""

from __future__ import annotations

import torch


def compute_dtype(cfg):
    device = str(cfg["device"])
    requested = cfg.get("compute_dtype", "auto")
    if requested not in {"auto", "float32", "bfloat16"}:
        raise ValueError("Proof compute_dtype must be auto, float32 or bfloat16")
    if device == "cpu":
        return torch.float32
    if not torch.cuda.is_available():
        raise ValueError("GPU unavailable")
    if torch.version.hip:
        # ROCm reports gfx versions as capability; ask for native BF16 directly.
        with torch.cuda.device(device):
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    else:
        native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    if requested == "bfloat16" and not native_bf16:
        raise ValueError("Selected GPU has no native BF16 support; use auto or float32")
    return torch.bfloat16 if native_bf16 and requested != "float32" else torch.float32
