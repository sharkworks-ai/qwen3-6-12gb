"""One precision policy shared by reconstruction, recovery and export."""

from __future__ import annotations

import hashlib
import json
import re


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor_spec(name: str, cfg: dict) -> dict | None:
    # Routers, normalisation, biases and state-space scalars stay full precision.
    if not re.search(r"(\.weight$|\.experts\.(gate_up_proj|down_proj)$)", name):
        return None
    if re.search(r"(norm|embed|lm_head|\.gate\.weight$|router|shared_expert_gate)", name):
        return None
    routed = ".experts." in name and "shared_expert" not in name
    if routed:
        bits = 2 if cfg["preset"] == "aggressive" else 1.58
        method = "gsq" if bits == 2 else "ayot_ternary_reference"
    elif "shared_expert" in name:
        bits, method = int(cfg.get("shared_bits", 4)), "gsq"
    else:
        bits, method = int(cfg.get("attention_bits", 4)), "gsq"
    for rule in cfg.get("sensitive_tensors", []):
        if re.search(rule["pattern"], name):
            bits, method = int(rule["bits"]), "gsq"
            break
    return {"bits": bits, "method": method, "group_size": int(cfg.get("group_size", 64))}


def validate(cfg: dict) -> None:
    if cfg.get("engine", "reference") not in {"reference", "upstream_window"}:
        raise ValueError("Unknown reconstruction engine")
    if int(cfg.get("checkpoint_steps", 25)) < 1:
        raise ValueError("checkpoint_steps must be positive")
    width = int(cfg.get("window_size", 1))
    if not 1 <= int(cfg.get("window_stride", 1)) <= width:
        raise ValueError("Require 1 <= window_stride <= window_size")
    if cfg.get("preset") not in {"aggressive", "extreme"}:
        raise ValueError("preset must be aggressive or extreme")
    if int(cfg.get("context_length", 262144)) != 262144:
        raise ValueError("Keep the native 262144-token context target")
    for key in ("attention_bits", "shared_bits"):
        if int(cfg.get(key, 4)) not in {3, 4}:
            raise ValueError(f"{key} must be 3 or 4")
    if int(cfg.get("group_size", 64)) not in {32, 64, 128}:
        raise ValueError("group_size must be 32, 64 or 128")
    for rule in cfg.get("sensitive_tensors", []):
        re.compile(rule["pattern"])
        if int(rule["bits"]) not in {2, 3}:
            raise ValueError("Sensitive tensors must use GSQ 2 or 3 bits")
    if cfg["preset"] == "extreme" and not cfg.get("sensitive_tensors"):
        raise ValueError("Extreme requires explicit sensitive tensor rules")
    if int(cfg.get("steps", 100)) < 1:
        raise ValueError("steps must be positive")


def build_map(names: list[str], cfg: dict) -> dict:
    validate(cfg)
    entries = {n: s for n in names if (s := tensor_spec(n, cfg)) is not None}
    if not any(".experts." in n for n in entries):
        raise ValueError("No routed expert tensors matched")
    for rule in cfg.get("sensitive_tensors", []):
        if not any(re.search(rule["pattern"], n) for n in entries):
            raise ValueError(f"Sensitivity rule matched no tensor: {rule['pattern']}")
    return {"schema": 1, "preset": cfg["preset"], "tensors": entries}
