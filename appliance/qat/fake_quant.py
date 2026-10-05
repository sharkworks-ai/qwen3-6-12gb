from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class FakeQuantSpec:
    bits: float
    group_size: int=64
    ternary: bool=False

def _groups(x: torch.Tensor,g:int):
    flat=x.reshape(-1); pad=(-flat.numel())%g
    if pad: flat=torch.nn.functional.pad(flat,(0,pad))
    return flat.view(-1,g),x.shape

def _restore(x,shape): return x.reshape(-1)[:int(torch.tensor(shape).prod().item())].reshape(shape)

def fake_quant_weight(w: torch.Tensor,s:FakeQuantSpec) -> torch.Tensor:
    x,shape=_groups(w,s.group_size)
    if s.ternary or s.bits<=1.6:
        scale=x.detach().abs().mean(-1,keepdim=True).clamp_min(1e-8); t=.7*scale
        dq=torch.where(x>t,scale,torch.where(x<-t,-scale,torch.zeros_like(x)))
    else:
        qmax=2**(int(s.bits)-1)-1; scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
        dq=(x/scale).round().clamp(-qmax,qmax)*scale
    return _restore(x+(dq-x).detach(),shape)
