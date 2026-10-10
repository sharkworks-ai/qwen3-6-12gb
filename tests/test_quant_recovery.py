from pathlib import Path

import torch

from appliance.qat.fake_quant import FakeQuantSpec, fake_quant_weight
from appliance.recovery.escalation import RecoveryPolicy, decide
from appliance.recovery.failure_buffer import FailureBuffer


def test_fake_quant_shape_and_grad():
    x=torch.randn(17,requires_grad=True); y=fake_quant_weight(x,FakeQuantSpec(3,8)); assert y.shape==x.shape; y.sum().backward(); assert x.grad is not None

def test_ternary_fake_quant():
    x=torch.tensor([-2.,-.01,0.,.01,2.],requires_grad=True); y=fake_quant_weight(x,FakeQuantSpec(1.58,5,True)); assert torch.count_nonzero(y).item()<=2

def test_escalation():
    ref={'coding_score':100,'agent_score':100,'long_context_score':100}
    assert decide({'coding_score':99,'agent_score':99,'long_context_score':99},ref,RecoveryPolicy())['action']=='accept_without_recovery'
    assert decide({'coding_score':97,'agent_score':97,'long_context_score':97},ref,RecoveryPolicy())['action']=='post_quant_recovery'
    assert decide({'coding_score':94,'agent_score':94,'long_context_score':97},ref,RecoveryPolicy())['action']=='qat_recovery'

def test_failure_buffer(tmp_path:Path):
    b=FailureBuffer(tmp_path/'f.jsonl'); rows=[{'prompt':'x','teacher_success':True,'student_success':False,'teacher_output':'good','student_output':'bad'}]
    assert b.extend_from_comparison(rows)==1; assert b.extend_from_comparison(rows)==0; assert len(b.read())==1

def test_low_rank_qat_trains_adapter_through_quantizer():
    from torch.nn.utils import parametrize

    from appliance.qat.parametrize import apply_fake_quant, remove_fake_quant

    experts = torch.nn.Module()
    experts.gate_up_proj = torch.nn.Parameter(torch.randn(4, 8, 64))
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = experts
    model.requires_grad_(False)
    base = experts.gate_up_proj.detach().clone()
    spec = FakeQuantSpec(3, 64)
    assert apply_fake_quant(model, "q3_moe", rank=2) == ["mlp.experts.gate_up_proj"]
    # Only the per-expert adapters train; B starts at zero, so training starts at fq(W).
    prefix = "mlp.experts.parametrizations.gate_up_proj.0."
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == [
        prefix + "lora_a",
        prefix + "lora_b",
    ]
    assert torch.equal(experts.gate_up_proj, fake_quant_weight(base, spec))
    adapter = experts.parametrizations.gate_up_proj[0]
    (experts.gate_up_proj * torch.randn_like(base)).sum().backward()
    assert adapter.lora_b.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.lora_b.add_(0.1)
    expected = fake_quant_weight(base + adapter.lora_b @ adapter.lora_a, spec)
    remove_fake_quant(model)
    assert not parametrize.is_parametrized(experts)
    assert torch.allclose(experts.gate_up_proj, expected)
