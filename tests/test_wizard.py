from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from appliance.stages.common import save_json
from appliance.wizard.config import compile_plan


def real_answers(tmp_path, goal="train"):
    model = tmp_path / "model"
    model.mkdir()
    save_json(
        model / "config.json",
        {"max_position_embeddings": 262144, "num_experts": 16, "num_experts_per_tok": 2},
    )
    result = {
        "goal": goal,
        "name": "real-test",
        "source_model": str(model),
        "prune": True,
        "keep_experts": 8,
    }
    for key in [
        "training_file",
        "calibration_file",
        "recovery_dataset",
        "heldout_file",
        "prompts_file",
    ]:
        path = tmp_path / f"{key}.jsonl"
        path.write_text(json.dumps({"text": key}) + "\n")
        result[key] = str(path)
    return result


def test_plan_compiles_only_supported_operations(tmp_path):
    proof = compile_plan(
        {"goal": "proof", "name": "tiny", "commands": ["bad"], "cpu_test": True}, str(tmp_path)
    )
    assert proof["config"]["device"] == "cuda:0" and "cpu_test" not in proof["config"]
    answers = real_answers(tmp_path)
    plan = compile_plan(answers, str(tmp_path))
    assert plan["stages"][:2] == ["SFT / QLoRA", "Merge adapter"]
    assert plan["config"]["variants"] == ["aggressive", "extreme"]
    assert plan["config"]["max_new_tokens"] + plan["config"]["context_lengths"][0] <= 262144
    assert plan["config"]["keep_experts"] == 8
    answers.update(goal="compress", variants=["aggressive"], prune=False)
    answers.pop("training_file")
    answers.pop("prompts_file")
    plan = compile_plan(answers, str(tmp_path))
    assert (
        "SFT / QLoRA" not in plan["stages"]
        and "Generate teacher reasoning calibration" not in plan["stages"]
    )


def test_plan_rejects_bad_paths_data_and_budget(tmp_path):
    answers = real_answers(tmp_path)
    for change, match in [
        ({"heldout_file": answers["training_file"]}, "separate"),
        ({"source_model": "/etc"}, "under"),
        ({"name": "../escape"}, "name"),
        ({"cuda_devices": "0,0"}, "distinct"),
        ({"keep_experts": 1}, "top-k"),
        ({"benchmark_profile": "shell"}, "not configured"),
        ({"context_length": 250000, "max_new_tokens": 16384}, "native context"),
    ]:
        with pytest.raises(ValueError, match=match):
            compile_plan({**answers, **change}, str(tmp_path))
    save_json(Path(answers["source_model"]) / "config.json", {"qwen12g_proof_model": True})
    with pytest.raises(ValueError, match="synthetic"):
        compile_plan(answers, str(tmp_path))
    with pytest.raises(ValueError, match="one GPU"):
        compile_plan({"goal": "proof", "cuda_devices": "0,1"}, str(tmp_path))


def test_web_wizard_plan_launch_and_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN12G_WEB_TOKEN", "unit-test-token-123")
    from fastapi.testclient import TestClient

    from appliance import app as module

    monkeypatch.setattr(module, "settings", replace(module.settings, data_root=tmp_path))
    monkeypatch.setattr(
        module, "gpu_info", lambda: [{"index": "0", "name": "Test GPU", "memory_total": "12288"}]
    )
    calls = []
    monkeypatch.setattr(module.jobs, "reconcile", lambda: None)
    monkeypatch.setattr(module.db, "list_jobs", lambda *args: [])
    monkeypatch.setattr(
        module.jobs, "start", lambda kind, cfg: calls.append((kind, cfg)) or "wizard-test"
    )
    headers = {"Authorization": "Bearer unit-test-token-123"}
    with TestClient(module.app) as client:
        assert client.get("/api/v1/wizard/plan").status_code == 401
        page = client.get("/", headers=headers)
        assert (
            page.status_code == 200
            and "Wizard steps" in page.text
            and "Start automated run" in page.text
        )
        answers = {"goal": "proof", "name": "wizard-test"}
        response = client.post("/api/v1/wizard/plan", json=answers, headers=headers)
        assert response.status_code == 200 and len(response.json()["stages"]) == 7
        launched = client.post("/api/v1/wizard/start", json=answers, headers=headers)
        assert launched.status_code == 200 and launched.json()["url"] == "/runs/wizard-test"
        assert calls[0][0] == "automated_run"
        monkeypatch.setattr(
            module.db,
            "get_job",
            lambda _: SimpleNamespace(kind="automated_run", config=calls[0][1]),
        )
        assert (
            '"resume": true' in client.get("/wizard?resume_run=wizard-test", headers=headers).text
        )
        monkeypatch.setattr(module, "gpu_info", list)
        assert client.post("/api/v1/wizard/start", json=answers, headers=headers).status_code == 422
        assert len(calls) == 1
        monkeypatch.setattr(module, "gpu_info", lambda: [{"index": "0"}])
        monkeypatch.setattr(
            module.db,
            "list_jobs",
            lambda *args: [
                SimpleNamespace(
                    status="running", kind="sft", config={"cuda_devices": "0"}, run_id="busy"
                )
            ],
        )
        response = client.post("/api/v1/wizard/start", json=answers, headers=headers)
        assert response.status_code == 422 and "busy" in response.json()["detail"]


def test_orchestrator_wires_stages_and_resume_integrity(monkeypatch, tmp_path):
    from appliance.proof import state
    from appliance.wizard import job

    answers = real_answers(tmp_path)
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    calls = []
    fail = {"value": True}

    def fake_process(command, **kwargs):
        worker = json.loads(Path(command[-1]).read_text())
        cfg = worker["job_config"]
        kind = worker["job_kind"]
        calls.append((kind, cfg))
        if kind == "merge" and fail["value"]:
            return SimpleNamespace(returncode=1)
        if kind == "full_validation":
            bundle = Path(cfg["output_dir"]) / "bundle"
            bundle.mkdir(parents=True)
            save_json(bundle / "mixed-manifest.json", {"synthetic_proof": False})
            result = {
                "status": "completed",
                "variants": {
                    "aggressive": {
                        "bundle": str(bundle),
                        "metrics": {"heldout_loss": 2.0},
                        "contexts": [
                            {
                                "needle_found": True,
                                "peak_allocated_mib": 1024,
                                "peak_device_mib": 1024,
                            }
                        ],
                    }
                },
                "checks": {},
                "production_runtime": False,
            }
            save_json(Path(worker["result_file"]), result)
        elif kind == "profile":
            save_json(Path(cfg["output"]), {"layers": {}})
            result = {"status": "completed"}
        elif kind == "ayot":
            Path(cfg["output_file"]).write_text("teacher trace")
            result = {"status": "completed"}
        else:
            save_json(Path(cfg["output_dir"]) / "stage-manifest.json", {"kind": kind})
            result = {"status": "completed"}
        save_json(Path(worker["stage_dir"]) / "result.json", result)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(state.subprocess, "run", fake_process)
    monkeypatch.setattr(state.NvidiaTelemetry, "start", lambda _: None)
    monkeypatch.setattr(state.NvidiaTelemetry, "stop", lambda _: None)
    with pytest.raises(RuntimeError, match="merge"):
        job.run({"answers": answers})
    output = tmp_path / "artifacts/wizard/real-test"
    assert json.loads((output / "proof-progress.json").read_text())["status"] == "failed"
    fail["value"] = False
    report = job.run({"answers": {**answers, "resume": True}})
    assert [kind for kind, cfg in calls] == [
        "sft",
        "merge",
        "merge",
        "profile",
        "prune",
        "ayot",
        "full_validation",
    ]
    validation_cfg = calls[-1][1]
    assert validation_cfg["source_model"] == str(output / "prune/model")
    assert validation_cfg["ayot_file"] == str(output / "ayot/traces.jsonl")
    assert report["selection"]["recommended"] is None
    count = len(calls)
    job.run({"answers": {**answers, "resume": True}})
    assert len(calls) == count
    (output / "prune/model/stage-manifest.json").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        job.run({"answers": {**answers, "resume": True}})


def test_worker_selects_registered_command_and_checkpoint(monkeypatch, tmp_path):
    from appliance.wizard import worker

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "checkpoint-50").mkdir()
    captured = []
    monkeypatch.setattr(
        worker.subprocess, "run", lambda command, **kwargs: captured.append(command)
    )
    cfg = {
        "stage_dir": str(tmp_path),
        "job_kind": "sft",
        "job_config": {"output_dir": str(adapter), "resume": True, "num_processes": 2},
    }
    worker.run(cfg)
    assert captured[0][0] == "torchrun"
    assert json.loads((tmp_path / "job-config.json").read_text())["resume_from_checkpoint"] is True
    with pytest.raises(ValueError, match="not allowed"):
        worker.run({**cfg, "job_kind": "arbitrary_shell"})


def test_automatic_harness_provenance_and_quality_gates(monkeypatch, tmp_path):
    from appliance.eval.oxcoder_gate import BASELINE
    from appliance.runtime.packed import sha256
    from appliance.wizard import benchmarks

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    save_json(bundle / "mixed-manifest.json", {"synthetic_proof": False})
    report = tmp_path / "report.json"
    save_json(report, {"status": "completed", "variants": {"aggressive": {"bundle": str(bundle)}}})
    profiles = tmp_path / "profiles.json"
    save_json(profiles, {"isolated": {"url": "https://harness.test/run", "revision": "pinned-v1"}})
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("QWEN12G_BENCHMARK_PROFILES", str(profiles))
    evidence = {
        "aggressive": {
            "artifact_manifest_sha256": sha256(bundle / "mixed-manifest.json"),
            "harness_revisions": {"profile": "pinned-v1"},
            "scores": {k: 100 for k in BASELINE},
        }
    }
    requests = []

    def service(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=evidence)

    actual_client = httpx.Client
    monkeypatch.setattr(
        benchmarks.httpx,
        "Client",
        lambda **kwargs: actual_client(transport=httpx.MockTransport(service), **kwargs),
    )
    config = {
        "report_file": str(report),
        "output_dir": str(tmp_path / "scores"),
        "benchmark_profile": "isolated",
        "request_id": "stable-run-id",
    }
    result = benchmarks.run(config)
    assert result["candidates"]["aggressive"]["gate"]["pass"]
    assert (
        requests[0]["request_id"] == "stable-run-id"
        and requests[0]["harness_revision"] == "pinned-v1"
    )
    evidence["aggressive"]["artifact_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        benchmarks.run(config)


def test_candidate_selection_requires_quality_and_measured_resources():
    from appliance.wizard.job import rank_candidates

    report = {
        "target_vram_gib": 11.5,
        "variants": {
            name: {
                "metrics": {"heldout_loss": loss},
                "bundle": name,
                "contexts": [
                    {"peak_allocated_mib": peak, "peak_device_mib": peak, "needle_found": True}
                ],
            }
            for name, loss, peak in [("aggressive", 2, 1000), ("extreme", 1, 14000)]
        },
    }
    assert rank_candidates(report)["recommended"] is None
    benchmark = {"candidates": {n: {"gate": {"pass": True}} for n in report["variants"]}}
    assert rank_candidates(report, benchmark)["recommended"] == "aggressive"
    report["variants"]["aggressive"]["contexts"][0]["peak_device_mib"] = None
    assert rank_candidates(report, benchmark)["recommended"] is None


def test_b_only_validation_requires_physical_memory_evidence(monkeypatch, tmp_path):
    from appliance.proof import validation

    answers = real_answers(tmp_path, goal="compress")
    config = compile_plan(
        {
            **answers,
            "variants": ["aggressive"],
            "prune": False,
            "cuda_devices": "7",
            "context_repeats": 1,
        },
        str(tmp_path),
    )["config"]
    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        validation, "load_components", lambda _: (None, None, None, {"revision": "pinned"})
    )
    measured = {"peak": 1000}
    called = []

    class FakeStages:
        def __init__(self, cfg, output, identity):
            self.path = output / "proof-progress.json"
            self.state = {"stages": {}}

        def run(self, name, operation, **options):
            called.append(name)
            self.state["stages"][name] = {
                "peak_vram_mib": {"7": measured["peak"]} if measured["peak"] is not None else {}
            }
            if operation == "context":
                return {"needle_found": True, "peak_allocated_mib": 900}
            return {"heldout_loss": 2.0}

    monkeypatch.setattr(validation, "Stages", FakeStages)
    report = validation.run(config)
    assert set(report["variants"]) == {"aggressive"}
    assert report["checks"]["within_vram_limit"]
    assert not any("extreme" in n for n in called)
    measured["peak"] = None
    report = validation.run(config)
    assert report["checks"]["within_vram_limit"] is False
