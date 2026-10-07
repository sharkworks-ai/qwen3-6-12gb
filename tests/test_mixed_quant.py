import json

import pytest
import torch
from safetensors.torch import save_file

from appliance.qat.parametrize import apply_precision_map, remove_fake_quant
from appliance.quant.lowbit import dequantize, pack_codes, reconstruct, unpack_codes
from appliance.quant.mixed_job import run
from appliance.quant.precision import build_map, tensor_spec
from appliance.quant.presets import preset


@pytest.mark.parametrize("bits,levels", [(1.58, 3), (2, 4), (3, 8), (4, 16)])
def test_packing_crosses_bytes_and_padding(bits, levels):
    codes = (torch.arange(71) % levels).byte()
    packed = pack_codes(codes, bits)
    assert torch.equal(unpack_codes(packed, bits, len(codes)), codes.long())
    assert len(packed) < len(codes)


def test_policy_matches_individual_and_fused_experts():
    individual = "model.layers.0.mlp.experts.3.down_proj.weight"
    fused = "model.layers.2.mlp.experts.gate_up_proj"
    shared = "model.layers.2.mlp.shared_expert.down_proj.weight"
    a, c = preset("aggressive"), preset("extreme")
    assert tensor_spec(individual, a)["bits"] == 2
    assert tensor_spec(individual, c)["bits"] == 3
    assert tensor_spec(fused, c)["bits"] == 1.58
    assert tensor_spec(shared, c)["bits"] == 4
    assert tensor_spec("model.layers.2.mlp.gate.weight", c) is None
    with pytest.raises(ValueError, match="matched no tensor"):
        build_map([fused], c)


def test_reconstruction_and_dequantization():
    w = torch.randn(3, 13)
    spec = {"bits": 2, "group_size": 8}
    codes, scales, error, shape = reconstruct(w, spec, steps=3)
    result = dequantize(pack_codes(codes, 2), scales, spec, shape)
    assert result.shape == w.shape
    assert torch.isfinite(result).all() and error >= 0
    assert (codes < 4).all()


def test_qat_exports_normal_parameter_keys():
    m = torch.nn.Sequential(torch.nn.Linear(7, 5, bias=False))
    precision = {"tensors": {"0.weight": {"bits": 2, "group_size": 8}}}
    assert apply_precision_map(m, precision) == ["0.weight"]
    m(torch.ones(2, 7)).sum().backward()
    assert m[0].parametrizations.weight.original.grad is not None
    remove_fake_quant(m)
    assert list(m.state_dict()) == ["0.weight"]
    with pytest.raises(ValueError):
        apply_precision_map(m, {"tensors": {"missing.weight": {"bits": 2}}})


def test_dry_plan_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_moe"}')
    save_file(
        {"model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(3, 4)},
        str(model / "model.safetensors"),
    )
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text":"code"}\n')
    cfg = {
        **preset("aggressive"),
        "source_model": str(model),
        "output_dir": str(tmp_path / "output"),
        "calibration_file": str(calibration),
    }
    result = run(cfg)
    assert result["status"] == "planned"
    assert not (tmp_path / "output" / "packed").exists()
    plan = json.loads((tmp_path / "output" / "plan.json").read_text())
    assert len(plan["identity"]) == 64
    cfg["output_dir"] = str(model)
    with pytest.raises(ValueError, match="separate"):
        run(cfg)


def test_web_preset_launch_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN12G_WEB_TOKEN", "unit-test-token-123")
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    from fastapi.testclient import TestClient

    from appliance.app import app, jobs

    calls = []
    monkeypatch.setattr(jobs, "start", lambda kind, cfg: calls.append((kind, cfg)) or "test-run")
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer unit-test-token-123"}
        response = client.get("/compression", headers=headers)
        assert response.status_code == 200 and "not a reproduction of ScaleQ" in response.text
        cfg = preset("extreme")
        response = client.post(
            "/jobs/start",
            headers=headers,
            follow_redirects=False,
            data={"kind": "mixed_pipeline", "config_json": json.dumps(cfg)},
        )
        assert response.status_code == 303 and calls[0][0] == "mixed_pipeline"
        cfg["recovery"]["enabled"] = False
        assert (
            client.post(
                "/jobs/start",
                headers=headers,
                data={"kind": "mixed_pipeline", "config_json": json.dumps(cfg)},
            ).status_code
            == 422
        )


def test_complete_cpu_artifact_export_resume_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_moe"}')
    name = "model.layers.0.mlp.experts.0.down_proj.weight"
    save_file(
        {name: torch.randn(3, 7), "model.norm.weight": torch.ones(7)},
        str(model / "model.safetensors"),
    )
    corpus = tmp_path / "calibration.jsonl"
    corpus.write_text('{"text":"code"}\n')

    def fake_moments(model, calibration, precision, cfg, output):
        save_file({name: torch.ones(7)}, str(output))

    monkeypatch.setattr("appliance.quant.moments.collect_moments", fake_moments)
    cfg = {
        **preset("aggressive"),
        "source_model": str(model),
        "output_dir": str(tmp_path / "output"),
        "calibration_file": str(corpus),
        "steps": 2,
        "device": "cpu",
        "dry_run": False,
    }
    result = run(cfg)
    assert result["runtime_ready"] is False
    research = tmp_path / "output" / "research-hf"
    index = json.loads((research / "model.safetensors.index.json").read_text())
    assert name in index["weight_map"] and "model.norm.weight" in index["weight_map"]
    cfg["resume"] = True
    assert run(cfg)["identity"] == result["identity"]
    cfg["steps"] = 3
    with pytest.raises(ValueError, match="Resume rejected"):
        run(cfg)
    cfg["steps"] = 2
    item = result["tensors"][name]
    (tmp_path / "output" / item["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run(cfg)


def test_eager_fused_qwen_moe_calibration(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from transformers import BatchEncoding, Qwen3MoeConfig, Qwen3MoeForCausalLM

    from appliance.quant.moments import collect_moments

    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
        )
    )

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }
            )

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text":"code"}\n')
    precision = build_map([n for n, _ in model.named_parameters()], preset("aggressive"))
    output = tmp_path / "moments.safetensors"
    collect_moments(tmp_path, corpus, precision, {}, output)
    result = load_file(str(output))
    assert result["model.layers.0.mlp.experts.down_proj"].shape == (8,)
    assert result["model.layers.0.mlp.experts.gate_up_proj"].shape == (16,)
    assert all(torch.isfinite(t).all() for t in result.values())
