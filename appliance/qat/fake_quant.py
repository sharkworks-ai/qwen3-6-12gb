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
