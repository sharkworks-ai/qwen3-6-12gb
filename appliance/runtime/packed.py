"""Bounded eager inference directly from mixed packed artifacts.

No GSQ, CAT-Q, QAT or reconstruction modules are imported by this runtime.
This is a correctness/memory implementation, not a fused CUDA production kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from appliance.gpu import compute_dtype
from safetensors import safe_open
from safetensors.torch import load_file


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def child(root, value):
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Artifact path escapes its bundle")
    return path


class PackedWeight(torch.nn.Module):
    def __init__(self, codes, scales, spec, shape, chunk_rows=128):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales.reshape(-1))
        self.bits = spec["bits"]
        self.group = int(spec["group_size"])
        self.shape = tuple(shape)
        self.chunk_rows = int(chunk_rows)
        if self.bits not in (1.58, 2, 3, 4) or self.chunk_rows < 1:
            raise ValueError("Unsupported bits or invalid decode chunk")
        self.columns = math.ceil(shape[-1] / self.group) * self.group
        count = math.prod(shape[:-1]) * self.columns
        expected = math.ceil(count / 5) if self.bits == 1.58 else math.ceil(count * self.bits / 8)
        if codes.dtype != torch.uint8 or codes.numel() != expected:
            raise ValueError("Invalid packed code length or dtype")
        if scales.numel() != count // self.group or not torch.isfinite(scales).all():
            raise ValueError("Invalid packed scales")
        if self.bits == 1.58 and (codes > 242).any():
            raise ValueError("Invalid base-3 code byte")

    def decode(self, first_row, rows, dtype):
        positions = torch.arange(
            first_row * self.columns, (first_row + rows) * self.columns, device=self.codes.device
        )
        if self.bits == 1.58:
            powers = torch.tensor([1, 3, 9, 27, 81], device=positions.device)
            values = (self.codes[positions // 5].long() // powers[positions % 5]) % 3 - 1
        else:
            bits = int(self.bits)
            offsets = positions * bits
            indices = offsets // 8
            lo = self.codes[indices].long() >> (offsets % 8)
            # Last crossing byte may be padding outside the serialized buffer.
            next_indices = indices + 1
            hi = self.codes[next_indices.clamp_max(self.codes.numel() - 1)].long()
            hi = torch.where(next_indices < self.codes.numel(), hi, 0)
            values = (lo | (hi << (8 - offsets % 8))) & ((1 << bits) - 1)
            values = values - (1 << (bits - 1))
        result = values.to(dtype) * self.scales[positions // self.group].to(dtype)
        return result.reshape(rows, self.columns)[:, : self.shape[-1]]

    def linear(self, inputs, expert=None):
        rows = self.shape[-2]
        first = 0 if expert is None else int(expert) * rows
        output = inputs.new_empty(*inputs.shape[:-1], rows)
        for start in range(0, rows, self.chunk_rows):
            count = min(rows - start, self.chunk_rows)
            weight = self.decode(first + start, count, inputs.dtype)
            output[..., start : start + count] = torch.nn.functional.linear(inputs, weight)
        return output


class PackedLinear(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.packed = weight
        self.bias = bias

    def forward(self, inputs):
        result = self.packed.linear(inputs)
        return result if self.bias is None else result + self.bias


class PackedExperts(torch.nn.Module):
    def __init__(self, original, gate_up, down):
        super().__init__()
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        self.gate_up = gate_up
        self.down = down

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in torch.unique(top_k_index).tolist():
            if expert == self.num_experts:
                continue
            if not 0 <= expert < self.num_experts:
                raise ValueError("Invalid routed expert index")
            tokens, slots = torch.where(top_k_index == expert)
            gate, up = self.gate_up.linear(hidden_states[tokens], expert).chunk(2, -1)
            values = self.down.linear(self.act_fn(gate) * up, expert)
            values *= top_k_weights[tokens, slots, None]
            result.index_add_(0, tokens, values.to(result.dtype))
        return result


def load_packed(bundle, *, device="cuda:0", dtype=None, chunk_rows=128):
    """Build on meta; never load quantized BF16 parameters from research shards."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = compute_dtype(device) if dtype is None else dtype
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "mixed-manifest.json").read_text())
    if manifest.get("status") != "succeeded" or not manifest.get("tensors"):
        raise ValueError("A complete mixed artifact is required")
    # Resolve bundled research files locally, not stale absolute producer paths.
    research = bundle / "research-hf"
    exports = manifest.get("research_files")
    if exports:
        for name, expected_hash in exports.items():
            if sha256(child(research, name)) != expected_hash:
                raise ValueError(f"Research export integrity failure: {name}")
    else:
        # Older artifacts hash committed block shards in window-progress.json.
        # They remain readable, but lack complete outer/tokenizer provenance.
        progress = json.loads((bundle / "window-progress.json").read_text())
        for name, expected_hash in progress.get("exports", {}).items():
            if name.startswith("research-hf/") and sha256(child(bundle, name)) != expected_hash:
                raise ValueError(f"Research export integrity failure: {name}")
    config = AutoConfig.from_pretrained(research, local_files_only=True, trust_remote_code=False)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager", trust_remote_code=False
        )
    targets = manifest["tensors"]
    expected = dict(model.named_parameters())
    if set(targets) - set(expected):
        raise ValueError("Packed tensor map does not match model architecture")
    packed = {}
    for name, item in targets.items():
        if tuple(item["shape"]) != tuple(expected[name].shape):
            raise ValueError(f"Packed shape mismatch: {name}")
        path = child(bundle, item["file"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Packed integrity failure: {name}")
        data = load_file(str(path))
        packed[name] = PackedWeight(
            data["codes"], data["scales"], item["spec"], item["shape"], chunk_rows
        )
    modules = dict(model.named_modules())
    replaced = set()
    for name, weight in packed.items():
        parent, parameter = name.rsplit(".", 1)
        original = modules[parent]
        if isinstance(original, torch.nn.Linear) and parameter == "weight":
            replacement = PackedLinear(weight, original.bias)
            model.set_submodule(parent, replacement)
            replaced.add(name)
        elif parameter == "gate_up_proj" and parent.endswith(".experts"):
            down_name = parent + ".down_proj"
            if down_name not in packed:
                raise ValueError("Both fused expert projections must be packed")
            model.set_submodule(parent, PackedExperts(original, weight, packed[down_name]))
            replaced.update((name, down_name))
    if replaced != set(packed):
        raise ValueError(f"Unsupported packed projection: {sorted(set(packed) - replaced)}")
    loaded = set()
    for path in sorted(research.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for name in stream.keys():  # noqa: SIM118 - safe_open is not a mapping
                if name in targets:
                    continue
                parent, parameter = name.rsplit(".", 1)
                module = model.get_submodule(parent)
                value = stream.get_tensor(name)
                if value.is_floating_point():
                    value = value.to(dtype)
                if parameter in module._parameters:
                    module._parameters[parameter] = torch.nn.Parameter(value, requires_grad=False)
                elif parameter in module._buffers:
                    module._buffers[parameter] = value
                else:
                    raise ValueError(f"Unknown outer parameter: {name}")
                loaded.add(name)
    missing = set(expected) - set(targets) - loaded
    if missing or any(p.is_meta for p in model.parameters()):
        raise ValueError(f"Missing outer parameters: {sorted(missing)[:8]}")
    model = model.to(device).eval().requires_grad_(False)
    return model
