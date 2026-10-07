"""Experimental reference algorithms, not upstream benchmark reproductions.

GSQ uses learnable discrete assignments and group scales with Gumbel-Softmax.
Ternary uses the same assignment relaxation on AYOT activation moments.
Packed tensors are research artifacts. They are NOT a vLLM/GGUF format.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def grid(bits: float, device) -> torch.Tensor:
    if bits == 1.58:
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    return torch.arange(-(2 ** (int(bits) - 1)), 2 ** (int(bits) - 1), device=device).float()


def reconstruct(weight, spec: dict, *, steps=100, lr=0.01, seed=42, importance=None):
    """Memory bounded per-row optimisation; RTN initialisation is explicit."""
    torch.manual_seed(seed)
    original_shape = weight.shape
    rows = weight.detach().float().reshape(-1, weight.shape[-1])
    group = spec["group_size"]
    padded = F.pad(rows, (0, (-rows.shape[-1]) % group))
    w = padded.reshape(-1, group)
    levels = grid(spec["bits"], w.device)
    scale0 = w.abs().amax(-1, keepdim=True).clamp_min(1e-8) / levels.abs().max()
    log_scale = torch.nn.Parameter(scale0.log())
    initial = (w.unsqueeze(-1) / scale0.unsqueeze(-1) - levels).square()
    logits = torch.nn.Parameter(-initial.clamp_max(30))
    optimizer = torch.optim.Adam([logits, log_scale], lr=lr)
    # Optional activation second moments weight the reconstruction error.
    if importance is None:
        importance = torch.ones_like(rows)
    else:
        importance = importance.to(rows).expand_as(rows)
    mask = F.pad(importance, (0, padded.shape[-1] - rows.shape[-1])).reshape_as(w)
    for step in range(steps):
        optimizer.zero_grad()
        tau = 2.0 * (0.25 ** (step / max(1, steps - 1)))
        assignment = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        dq = (assignment * levels).sum(-1) * log_scale.exp()
        loss = ((dq - w).square() * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        codes = logits.argmax(-1).to(torch.uint8)
        scales = log_scale.exp().to(torch.float16)
        dq = levels[codes.long()] * scales.float()
        error = float(((dq - w).square() * mask).sum() / mask.sum().clamp_min(1))
    return codes.cpu(), scales.cpu(), error, list(original_shape)


def pack_codes(codes: torch.Tensor, bits: float) -> torch.Tensor:
    flat = codes.flatten().to(torch.int64)
    if bits == 1.58:
        # Five base-3 digits per byte, 1.6 stored bits/value before scales.
        flat = F.pad(flat, (0, (-len(flat)) % 5)).reshape(-1, 5)
        return (flat * torch.tensor([1, 3, 9, 27, 81])).sum(-1).to(torch.uint8)
    width = int(bits)
    positions = torch.arange(len(flat)) * width
    packed = torch.zeros(math.ceil(len(flat) * width / 8) + 1, dtype=torch.int64)
    packed.scatter_add_(0, positions // 8, (flat << (positions % 8)) & 255)
    packed.scatter_add_(0, positions // 8 + 1, flat >> (8 - positions % 8))
    return packed[:-1].to(torch.uint8)


def unpack_codes(packed, bits: float, count: int):
    if bits == 1.58:
        return ((packed.long()[:, None] // torch.tensor([1, 3, 9, 27, 81])) % 3).flatten()[:count]
    width = int(bits)
    data = F.pad(packed.long(), (0, 1))
    positions = torch.arange(count) * width
    return (
        (data[positions // 8] >> (positions % 8))
        | (data[positions // 8 + 1] << (8 - positions % 8))
    ) & ((1 << width) - 1)


def dequantize(packed, scales, spec: dict, shape: list[int]):
    rows = math.prod(shape[:-1])
    columns = math.ceil(shape[-1] / spec["group_size"]) * spec["group_size"]
    codes = unpack_codes(packed, spec["bits"], rows * columns).reshape(-1, spec["group_size"])
    values = grid(spec["bits"], "cpu")[codes] * scales.float()
    return values.reshape(rows, columns)[:, : shape[-1]].reshape(shape)
