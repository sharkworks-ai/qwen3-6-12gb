import json
from pathlib import Path

import pytest

from appliance.quant.base import QuantContext
from appliance.quant.bittern import BitTernBackend
from appliance.quant.registry import BACKENDS, get_backend


def test_backend_registry():
    assert {"llama_cpp", "turboquant", "bittern_catq"} <= set(BACKENDS)


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend("missing")


def test_bittern_probe_qwen36(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )
    bittern = tmp_path / "BitTern"
    (bittern / "projects" / "cat-q").mkdir(parents=True)

    ctx = QuantContext(
        source_model=model,
        output_dir=tmp_path / "out",
        config={"bittern_root": str(bittern)},
    )
    probe = BitTernBackend().probe(ctx)
    assert probe["compatible"] is True
    assert probe["experimental_qwen36_adapter_required"] is True


def test_turboquant_runtime_command(tmp_path: Path):
    backend = get_backend("turboquant")
    cmd = backend.runtime_command(
        tmp_path / "model.gguf",
        {
            "turboquant_root": "/tq",
            "cache_k": "q8_0",
            "cache_v": "turbo3",
            "context": 262144,
        },
    )
    assert "turbo3" in cmd
    assert "262144" in cmd
