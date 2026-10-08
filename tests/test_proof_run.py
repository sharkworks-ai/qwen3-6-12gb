"""CPU integration harness for the complete miniature proof workflow."""

import json
import os
from pathlib import Path

import pytest
import torch

from appliance.proof.config import defaults
from appliance.proof.job import run
from appliance.quant.lowbit import dequantize, pack_codes
from appliance.runtime.packed import PackedWeight, load_packed


@pytest.mark.parametrize("bits", [2, 3, 4, 1.58])
def test_chunk_decode_and_selected_expert(bits):
    torch.manual_seed(2)
    shape = [3, 7, 35]
    group = 32
    count = 3 * 7 * 64
    codes = torch.randint(0, 3 if bits == 1.58 else 2**bits, (count,))
    scales = torch.rand(count // group, 1).half() + 0.01
    if bits != 1.58:
        scales[::2] *= -1
    spec = {"bits": bits, "group_size": group}
    packed = pack_codes(codes.byte(), bits)
    module = PackedWeight(packed, scales, spec, shape, chunk_rows=2)
    x = torch.randn(4, 35)
    expected = dequantize(packed, scales, spec, shape)
    for expert in range(3):
        assert torch.allclose(
            module.linear(x, expert), torch.nn.functional.linear(x, expected[expert]), atol=1e-5
        )


def test_complete_proof_and_resume(tmp_path, monkeypatch):
    gsq = Path(os.environ.get("QWEN12G_TEST_GSQ_ROOT", "/opt/GSQ"))
    catq = Path(os.environ.get("QWEN12G_TEST_CATQ_ROOT", "/opt/BitTern"))
    if not gsq.exists() or not catq.exists():
        pytest.skip("Pinned quantizer checkouts required")
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    cfg = {
        **defaults(),
        "output_dir": str(tmp_path / "proof"),
        "device": "cpu",
        "cpu_test": True,
        "hidden_size": 64,
        "layers": 3,
        "experts": 2,
        "expert_width": 64,
        "sequence_length": 32,
        "samples": 1,
        "train_steps": 2,
        "qat_steps": 2,
        "reconstruct_steps": 2,
        "checkpoint_steps": 5,
        "gsq_root": str(gsq),
        "bittern_root": str(catq),
    }
    result = run(cfg)
    assert result["status"] == "completed"
    assert result["checks"]["packed_export_parity"]
    assert result["checks"]["checkpoint_restart"]
    assert result["pass"] is False  # No GPU measurement in CPU CI.
    assert result["ayot_validation"] == "not_tested_synthetic_fixture"
    assert result["model"]["native_context"] == 262144
    assert run({**cfg, "resume": True})["identity"] == result["identity"]
    torch.set_num_threads(1)
    model = load_packed(result["variants"]["extreme"]["bundle"], device="cpu", dtype=torch.float32)
    assert not any("experts.gate_up_proj" in n for n, _ in model.named_parameters())
    assert any("gate_up.codes" in n for n, _ in model.named_buffers())
    with torch.inference_mode():
        assert (
            model.generate(torch.tensor([[4, 5, 6]]), max_new_tokens=2, do_sample=False).shape[-1]
            == 5
        )
    artifact = Path(result["variants"]["extreme"]["bundle"])
    manifest = json.loads((artifact / "mixed-manifest.json").read_text())
    packed_file = artifact / next(iter(manifest["tensors"].values()))["file"]
    packed_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        load_packed(artifact, device="cpu")
    with pytest.raises(ValueError, match="integrity"):
        run({**cfg, "resume": True})


def test_proof_web_controls_and_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN12G_WEB_TOKEN", "unit-test-token-123")
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from appliance import app as module

    monkeypatch.setattr(module, "settings", replace(module.settings, data_root=tmp_path))
    calls = []
    monkeypatch.setattr(
        module.jobs, "start", lambda kind, cfg: calls.append((kind, cfg)) or "proof-test"
    )
    headers = {"Authorization": "Bearer unit-test-token-123"}
    with TestClient(module.app) as client:
        page = client.get("/proof", params={"output_dir": str(tmp_path / "proof")}, headers=headers)
        assert page.status_code == 200
        assert "Packed inference test" in page.text and "Full-model validation" in page.text
        cfg = {**defaults(), "output_dir": str(tmp_path / "proof")}
        response = client.post(
            "/jobs/start",
            headers=headers,
            follow_redirects=False,
            data={"kind": "proof_run", "config_json": json.dumps(cfg)},
        )
        assert response.status_code == 303 and calls[-1][0] == "proof_run"
        cfg["cpu_test"] = True
        assert (
            client.post(
                "/jobs/start",
                headers=headers,
                data={"kind": "proof_run", "config_json": json.dumps(cfg)},
            ).status_code
            == 422
        )
        report = {
            "status": "completed",
            "synthetic": True,
            "production_runtime": False,
            "checks": {"vram_measured": False},
        }
        (tmp_path / "proof").mkdir()
        (tmp_path / "proof" / "report.json").write_text(json.dumps(report))
        assert (
            "Not passed"
            in client.get(
                "/proof", params={"output_dir": str(tmp_path / "proof")}, headers=headers
            ).text
        )
        assert (
            "must stay under"
            in client.get("/proof", params={"output_dir": "/etc"}, headers=headers).text
        )


def test_benchmark_evidence_binding(tmp_path):
    from appliance.eval.oxcoder_gate import BASELINE
    from appliance.proof.benchmarks import compare
    from appliance.runtime.packed import sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "mixed-manifest.json"
    manifest.write_text('{"synthetic_proof":false}')
    report = {"status": "completed", "variants": {"aggressive": {"bundle": str(bundle)}}}
    evidence = {
        "aggressive": {
            "artifact_manifest_sha256": sha256(manifest),
            "harness_revisions": {"test": "pinned"},
            "scores": {k: 100 for k in BASELINE},
        }
    }
    assert compare(report, evidence, str(tmp_path))["candidates"]["aggressive"]["gate"]["pass"]
    evidence["aggressive"]["artifact_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        compare(report, evidence, str(tmp_path))
    with pytest.raises(ValueError, match="non-synthetic"):
        compare({**report, "synthetic": True}, evidence, str(tmp_path))
