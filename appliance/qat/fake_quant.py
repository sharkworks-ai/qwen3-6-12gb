from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FakeQuantSpec:
    bits: float
    group_size: int = 64
    ternary: bool = False


def _groups(x: torch.Tensor, g: int):
    flat = x.reshape(-1)
    pad = (-flat.numel()) % g
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.view(-1, g), x.shape


def _restore(x, shape):
    return x.reshape(-1)[: int(torch.tensor(shape).prod().item())].reshape(shape)


def levels(s: FakeQuantSpec) -> tuple[int, int]:
    """Integer code range of the packed runtime grid (appliance.quant.lowbit.grid)."""
    if s.ternary or s.bits <= 1.6:
        return -1, 1
    return -(2 ** (int(s.bits) - 1)), 2 ** (int(s.bits) - 1) - 1


def fake_quant_on_grid(w: torch.Tensor, s: FakeQuantSpec, scales: torch.Tensor) -> torch.Tensor:
    """Round onto a reconstruction's fixed grid: per-group scales times integer levels.

    Weights already on that grid pass through unchanged, so QAT starts from the
    reconstructed model rather than re-rounding it onto a min/max grid.
    """
    x, shape = _groups(w, s.group_size)
    if scales.numel() != x.shape[0]:
        raise ValueError(f"Grid has {scales.numel()} scales for {x.shape[0]} groups")
    # FP32 division, so BF16 weights already on the grid round back to their level.
    # GSQ scales are signed, and a zero scale decodes its whole group to zero.
    scale = scales.reshape(-1, 1).float()
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    lo, hi = levels(s)
    # In place, so a fused expert tensor needs one FP32 temporary rather than several.
    dq = x.to(torch.float32, copy=True).div_(divisor).round_().clamp_(lo, hi).mul_(scale)
    dq = dq.to(x.dtype)
    return _restore(x + (dq - x).detach(), shape)


def fake_quant_weight(w: torch.Tensor, s: FakeQuantSpec) -> torch.Tensor:
    x, shape = _groups(w, s.group_size)
    if s.ternary or s.bits <= 1.6:
        scale = x.detach().abs().mean(-1, keepdim=True).clamp_min(1e-8)
        t = 0.7 * scale
        dq = torch.where(x > t, scale, torch.where(x < -t, -scale, torch.zeros_like(x)))
    else:
        qmin = -(2 ** (int(s.bits) - 1))
        qmax = 2 ** (int(s.bits) - 1) - 1
        scale = torch.maximum(
            x.detach().amax(-1, keepdim=True) / qmax, x.detach().amin(-1, keepdim=True) / qmin
        ).clamp_min(1e-8)
        dq = (x / scale).round().clamp(qmin, qmax) * scale
    return _restore(x + (dq - x).detach(), shape)
