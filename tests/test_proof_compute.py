import pytest
import torch

from appliance.proof.compute import compute_dtype
from appliance.wizard.config import compile_plan


@pytest.mark.parametrize(
    "capability,expected", [((7, 5), torch.float32), ((12, 0), torch.bfloat16)]
)
def test_proof_native_arithmetic(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert compute_dtype({"device": "cuda:0"}) == expected
    assert compute_dtype({"device": "cuda:0", "compute_dtype": "float32"}) == torch.float32
    assert compute_dtype({"device": "cpu"}) == torch.float32


def test_unsupported_bf16_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="no native BF16"):
        compute_dtype({"device": "cuda:0", "compute_dtype": "bfloat16"})


def test_laptop_wizard_rejects_arbitrary_size_and_keeps_native_context(tmp_path):
    plan = compile_plan({"goal": "proof", "proof_size": "laptop"}, str(tmp_path))
    assert plan["config"]["layers"] == 5
    assert plan["config"]["sequence_length"] == 64
    assert "cpu_test" not in plan["config"]
    with pytest.raises(ValueError, match="proof size"):
        compile_plan({"goal": "proof", "proof_size": "giant"}, str(tmp_path))


def test_proof_arithmetic_on_rocm_uses_native_bf16_query(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "device", lambda device: __import__("contextlib").nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)
    assert compute_dtype({"device": "cuda:0"}) == torch.float32
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    assert compute_dtype({"device": "cuda:0"}) == torch.bfloat16
