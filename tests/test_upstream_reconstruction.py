"""CPU functional checks of real pinned components; CUDA RNG is mocked only here."""

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from appliance.quant.lowbit import dequantize
from appliance.quant.mixed_job import file_hash
from appliance.quant.presets import preset
from appliance.quant.upstream import load_components
from appliance.quant.upstream_params import ProjectionQuantizer, typical_step_for_zero_groups
from appliance.quant.window_job import (
    assign_devices,
    check_window_fits,
    quantizer_cost,
    run,
    sample_order,
    windows,
)


@pytest.fixture
def upstream(monkeypatch):
    gsq = Path(os.environ.get("QWEN12G_TEST_GSQ_ROOT", "/opt/GSQ"))
    catq = Path(os.environ.get("QWEN12G_TEST_CATQ_ROOT", "/opt/BitTern"))
    if not gsq.exists() or not catq.exists():
        pytest.skip("Pinned source checkouts required")
    # Pinned GSQ autograd is CUDA-only. These substitutions let CPU CI exercise
    # the unchanged arithmetic/backward and deterministic RNG replay.
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda **k: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, **k: torch.set_rng_state(state))
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **k: original_fork(devices=[]))
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield {"gsq_root": str(gsq), "bittern_root": str(catq)}
    torch.set_num_threads(original_threads)


def test_window_overlap_and_validation():
    assert list(windows(5, 3, 1)) == [(0, 3, 1), (1, 4, 2), (2, 5, 5)]
    with pytest.raises(ValueError):
        list(windows(5, 2, 3))


@pytest.mark.parametrize("name", ["aggressive", "extreme"])
def test_pipeline_routes_upstream_engine_through_recovery(tmp_path, monkeypatch, name):
    from appliance.quant.pipeline import run as pipeline

    calls = []

    def reconstruct(cfg):
        calls.append(cfg.copy())
        return {
            "precision_map": str(tmp_path / "precision-map.json"),
            "research_model": str(tmp_path / "research-hf"),
        }

    def qat(command, check):
        assert check and "appliance.qat.job" in command

    monkeypatch.setattr("appliance.quant.window_job.run", reconstruct)
    monkeypatch.setattr("appliance.quant.pipeline.subprocess.run", qat)
    cfg = preset(name)
    cfg["dry_run"] = False
    cfg["recovery"]["enabled"] = True
    pipeline(cfg)
    assert len(calls) == 2
    assert calls[1]["source_model"] == str(tmp_path / "qat" / "recovered")
    assert calls[1]["engine"] == "upstream_window"
    if name == "extreme":
        assert calls[1]["teacher_model"] == cfg["source_model"]


@pytest.mark.parametrize("bits", [2, 3, 4, 1.58])
def test_real_components_gradients_and_packed_grid(upstream, bits):
    components = load_components({**upstream, "preset": "extreme"})
    q = ProjectionQuantizer(
        torch.randn(8, 32), {"bits": bits, "group_size": 32}, components, torch.eye(32), {}
    )
    q().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q.parameters())
    if bits == 1.58:
        assert q.lora_b.grad.abs().sum() > 0
    packed = q.packed()
    unpacked = dequantize(packed["codes"], packed["scales"], q.spec, list(q.shape))
    assert torch.allclose(unpacked, q().detach(), atol=0.002, rtol=0.002)


@pytest.mark.parametrize("name", ["aggressive", "extreme"])
@pytest.mark.parametrize("architecture", ["qwen3", "hybrid"])
def test_actual_moe_window_training_and_resume(upstream, tmp_path, monkeypatch, name, architecture):
    from transformers import BatchEncoding, Qwen3MoeConfig, Qwen3MoeForCausalLM

    monkeypatch.setenv("QWEN12G_DATA_ROOT", str(tmp_path))
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
    ).float()
    if architecture == "hybrid":
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

        model = Qwen3_5MoeForCausalLM(
            Qwen3_5MoeTextConfig(
                vocab_size=32,
                hidden_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                num_experts=2,
                num_experts_per_tok=2,
                moe_intermediate_size=32,
                shared_expert_intermediate_size=32,
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
            )
        ).float()
    source = tmp_path / "source"
    model.save_pretrained(source)

    class Tokenizer:
        def __call__(self, *a, **k):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                }
            )

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: copy.deepcopy(model)
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"coding trace"}\n')
    corpus.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "kind": "ayot",
                "teacher_model": str(source),
                "sha256": file_hash(corpus),
            }
        )
    )
    cfg = {
        **preset(name),
        **upstream,
        "source_model": str(source),
        "output_dir": str(tmp_path / "output"),
        "calibration_file": str(corpus),
        "device": "cpu",
        "window_devices": ["cpu"],
        "cpu_test": True,
        "group_size": 32,
        "steps": 2,
        "max_samples": 1,
        "checkpoint_steps": 1,
        "dry_run": False,
        "sensitive_tensors": [{"pattern": r"\.layers\.0\..*\.experts\.", "bits": 3}],
    }
    result = run(cfg)
    assert result["status"] == "succeeded" and result["runtime_ready"] is False
    assert result["implementation"] == "pinned_gsq_catq_sliding_adapter"
    # Resume-only state is removed once the reconstruction succeeds.
    for part in ("teacher-cache", "student-inputs", "training-checkpoints"):
        assert not (tmp_path / "output" / part).exists()
    if name == "extreme":
        assert any(x["spec"]["bits"] == 1.58 for x in result["tensors"].values())
    loaded = type(model).from_pretrained(result["research_model"])
    assert torch.isfinite(loaded(torch.tensor([[1, 2, 3]])).logits).all()
    # Interrupt after the first durable optimizer checkpoint, then verify exact
    # packed weights against an uninterrupted run, including overlapping windows.
    from lion_pytorch import Lion

    original_step = Lion.step
    calls = 0

    def interrupt_step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated worker interruption")
        return original_step(self, *args, **kwargs)

    interrupted_cfg = {**cfg, "output_dir": str(tmp_path / "interrupted")}
    monkeypatch.setattr(Lion, "step", interrupt_step)
    with pytest.raises(RuntimeError, match="simulated worker"):
        run(interrupted_cfg)
    monkeypatch.setattr(Lion, "step", original_step)
    resumed = run({**interrupted_cfg, "resume": True})
    assert {n: t["sha256"] for n, t in resumed["tensors"].items()} == {
        n: t["sha256"] for n, t in result["tensors"].items()
    }
    cfg["resume"] = True
    # Clean-up interrupted after the pruned record was saved: resume finishes it.
    (tmp_path / "output" / "teacher-cache").mkdir()
    assert run(cfg)["identity"] == result["identity"]
    assert not (tmp_path / "output" / "teacher-cache").exists()
    assert result["training_coverage"]["samples_trained_per_window"] == 1
    cfg["window_stride"] = 2
    with pytest.raises(ValueError, match="mismatch|Require"):
        run(cfg)


def test_zero_groups_get_the_typical_step_not_gptq_fallback():
    # Re-quantizing a quantized tensor: many groups rounded to zero. GPTQ gives those
    # the range [-1, 1] (step about 0.67 at 2 bits), so any GSQ code flip there became
    # a weight around 50x too large and destroyed the requantized model.
    matrix = torch.randn(4, 256) * 0.01
    matrix[0, :64] = 0
    matrix[2, 128:192] = 0
    scales = torch.full((4, 4), 0.01)
    scales[1, 0] = -0.02  # GSQ scales are signed
    scales[0, 0] = scales[2, 2] = 2 / 3  # GPTQ's zero-range fallback
    zero = torch.zeros(4, 4, dtype=torch.bool)
    zero[0, 0] = zero[2, 2] = True
    fixed = typical_step_for_zero_groups(matrix, scales, 64)
    assert fixed[0, 0] == fixed[2, 2] == scales[~zero].abs().median()
    assert torch.equal(fixed[~zero], scales[~zero])
    assert typical_step_for_zero_groups(torch.randn(4, 256), scales, 64) is scales


def test_window_quantizers_spread_across_gpus_by_size():
    # One block per window: its expert quantizers dominate memory and used to share the
    # block's GPU with everything else, peaking at 32 GB while the other GPU sat idle.
    two_bit, four_bit = {"bits": 2}, {"bits": 4}
    costs = {
        "experts.gate_up_proj": quantizer_cost(268_000_000, two_bit),
        "experts.down_proj": quantizer_cost(134_000_000, two_bit),
        "self_attn.q_proj": quantizer_cost(8_000_000, four_bit),
        "shared_expert.up_proj": quantizer_cost(1_000_000, four_bit),
    }
    placement, estimated = assign_devices(costs, {"cuda:0": 900_000_000}, ["cuda:0", "cuda:1"])
    assert placement["experts.gate_up_proj"] == "cuda:1"
    assert placement["experts.down_proj"] == "cuda:0"
    load = {"cuda:0": 900_000_000, "cuda:1": 0}
    for name, device in placement.items():
        load[device] += costs[name]
    assert max(load.values()) < 0.7 * sum(load.values())
    assert estimated == load
    assert set(assign_devices(costs, {}, ["cpu"])[0].values()) == {"cpu"}


def test_window_memory_check_rejects_oversized_windows(monkeypatch):
    # Extreme's first 2-block window with 3-bit sensitive experts needed about 43 GiB per
    # 32 GB GPU and crashed with an OOM while building quantizers; fail up front instead.
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=32 * 2**30)
    )
    check_window_fits({"cuda:0": 25 * 2**30, "cuda:1": 25 * 2**30}, 0, 2)
    check_window_fits({"cpu": 10**15}, 0, 2)
    with pytest.raises(ValueError, match=r"Window 0:2 needs about 43 GiB on cuda:1"):
        check_window_fits({"cuda:0": 20 * 2**30, "cuda:1": 43 * 2**30}, 0, 2)


def test_ternary_quantizer_returns_the_source_dtype(upstream):
    # CAT-Q computes in FP32; on GPUs the block runs in BF16, and the first ternary
    # window of a real extreme run failed with "BFloat16 != float" in the expert matmul.
    components = load_components({**upstream, "preset": "extreme"})
    weight = torch.randn(2, 8, 32).to(torch.bfloat16)
    q = ProjectionQuantizer(weight, {"bits": 1.58, "group_size": 32}, components, None, {})
    assert q().dtype == torch.bfloat16
    q().float().square().mean().backward()
    assert q.lora_b.grad.abs().sum() > 0


def test_reconstruction_sample_order_covers_the_set_reproducibly():
    # Cycling from 0 with 100 steps over 128 samples never trained on the last 28.
    order = sample_order(128, 42)
    assert sorted(order) == list(range(128)) and order != list(range(128))
    assert sample_order(128, 42) == order and sample_order(128, 43) != order


def test_needle_prompts_vary_position_secret_and_include_decoys():
    from appliance.proof.validation import needle_prompt

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    prompts = [needle_prompt(Tokenizer(), 6000, r) for r in range(3)]
    assert all(len(tokens) == 6000 for tokens, _, _ in prompts)
    assert len({secret for _, secret, _ in prompts}) == 3
    assert [position for _, _, position in prompts] == [0.1, 0.5, 0.9]
    tokens, secret, _ = prompts[0]
    text = "".join(chr(t) for t in tokens)
    assert text.count(f"current secret code is {secret}") == 1
    assert text.count("A retired code was") == 4
