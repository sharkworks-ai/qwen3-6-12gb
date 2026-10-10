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

def test_qat_trains_on_the_reconstruction_grid(tmp_path: Path):
    import json

    from safetensors.torch import save_file

    from appliance.qat.fake_quant import fake_quant_on_grid
    from appliance.qat.parametrize import (
        apply_precision_map,
        load_reconstruction_grids,
        remove_fake_quant,
    )

    torch.manual_seed(0)
    spec = FakeQuantSpec(4, 64)
    # A reconstructed 4-bit tensor: learned per-group scales times integer levels. GSQ's
    # MSE-fitted scales often leave the extreme levels (-8, 7) unused.
    scales = torch.rand(4 * 8 * 2) + 0.5
    codes = torch.randint(-3, 4, (4 * 8 * 2, 64)).float()
    on_grid = (codes * scales[:, None]).reshape(4, 8, 128)
    assert torch.equal(fake_quant_on_grid(on_grid, spec, scales), on_grid)
    # Min/max then picks a smaller step and re-rounds every weight off that grid.
    assert not torch.allclose(fake_quant_weight(on_grid, spec), on_grid)
    # GSQ scales are signed, and a zero scale decodes its group to zero, as at runtime.
    signed = scales.clone()
    signed[::2] *= -1
    signed[1] = 0
    flat = (codes * signed[:, None]).reshape(4, 8, 128)
    result = fake_quant_on_grid(flat, spec, signed)
    assert torch.isfinite(result).all() and torch.equal(result, flat)

    name = "mlp.experts.gate_up_proj"
    (tmp_path / "packed").mkdir()
    save_file({"codes": torch.zeros(1, dtype=torch.uint8), "scales": scales.half()[:, None]},
              str(tmp_path / "packed" / "t.safetensors"))
    (tmp_path / "mixed-manifest.json").write_text(
        json.dumps({"tensors": {name: {"file": "packed/t.safetensors"}}})
    )
    grids = load_reconstruction_grids(tmp_path)
    assert set(grids) == {name}

    experts = torch.nn.Module()
    experts.gate_up_proj = torch.nn.Parameter(on_grid.clone())
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = experts
    model.requires_grad_(False)
    precision = {"tensors": {name: {"bits": 4, "group_size": 64}}}
    assert apply_precision_map(model, precision, rank=2, grids=grids) == [name]
    # QAT starts from the reconstruction itself (FP16-stored scales round back exactly).
    assert torch.allclose(experts.gate_up_proj, on_grid, rtol=1e-3)
    adapter = experts.parametrizations.gate_up_proj[0]
    with torch.no_grad():
        adapter.lora_b.add_(0.05)
    expected = fake_quant_on_grid(on_grid + adapter.lora_b @ adapter.lora_a, spec, adapter.grid)
    remove_fake_quant(model)
    assert torch.allclose(experts.gate_up_proj, expected)
