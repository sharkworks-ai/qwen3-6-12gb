from __future__ import annotations

from pathlib import Path

from .base import QuantBackend, QuantContext, QuantResult
from .utils import run


class TurboQuantBackend(QuantBackend):
    """TurboQuant weight and KV quantization through the public llama.cpp fork."""

    name = "turboquant"
    experimental = True

    def probe(self, context: QuantContext) -> dict:
        root = Path(context.config.get("turboquant_root", "/opt/llama.cpp-turboquant"))
        quant = root / "build/bin/llama-quantize"
        server = root / "build/bin/llama-server"
        convert = root / "convert_hf_to_gguf.py"
        return {
            "compatible": quant.exists() and server.exists() and convert.exists(),
            "experimental": True,
            "turboquant_root": str(root),
            "supported_weight_types": [
                "TQ1_0",
                "TQ2_0",
                "TQ3_1S",
                "TQ3_4S",
                "TQ4_1S",
            ],
            "supported_kv_types": ["turbo2", "turbo3", "turbo4"],
        }

    def run(self, context: QuantContext) -> QuantResult:
        cfg = context.config
        root = Path(cfg.get("turboquant_root", "/opt/llama.cpp-turboquant"))
        context.output_dir.mkdir(parents=True, exist_ok=True)

        f16 = context.output_dir / "model-f16.gguf"
        run(
            [
                "python3",
                str(root / "convert_hf_to_gguf.py"),
                str(context.source_model),
                "--outfile",
                str(f16),
                "--outtype",
                cfg.get("convert_type", "f16"),
            ]
        )

        imatrix = None
        if cfg.get("use_imatrix", True):
            if context.calibration_file is None:
                raise ValueError("TurboQuant imatrix quantization requires calibration_file")
            imatrix = context.output_dir / "imatrix.dat"
            run(
                [
                    str(root / "build/bin/llama-imatrix"),
                    "-m",
                    str(f16),
                    "-f",
                    str(context.calibration_file),
                    "-o",
                    str(imatrix),
                    "-c",
                    str(cfg.get("imatrix_context", 512)),
                    "-ngl",
                    str(cfg.get("imatrix_gpu_layers", 999)),
                ]
            )

        weight_type = cfg.get("weight_type", "TQ2_0")
        output = context.output_dir / f"model-{weight_type}.gguf"
        args = [str(root / "build/bin/llama-quantize")]
        if imatrix is not None:
            args += ["--imatrix", str(imatrix)]

        # Mixed MoE policies can protect the dense path while pushing experts lower.
        for override in cfg.get("tensor_types", []):
            args += ["--tensor-type", override]

        args += [str(f16), str(output), weight_type]
        run(args)

        return QuantResult(
            backend=self.name,
            status="succeeded",
            artifacts=[str(output)],
            runtime="TurboQuant llama.cpp fork",
            notes=[
                "Artifact requires a TurboQuant-capable llama.cpp fork.",
                "TurboQuant KV is selected at inference time, separately from weight quantization.",
            ],
        )

    def runtime_command(self, artifact: Path, config: dict) -> list[str]:
        root = Path(config.get("turboquant_root", "/opt/llama.cpp-turboquant"))
        kv_k = config.get("cache_k", "q8_0")
        kv_v = config.get("cache_v", "turbo3")
        return [
            str(root / "build/bin/llama-server"),
            "--model",
            str(artifact),
            "--no-mmap",
            "--n-gpu-layers",
            str(config.get("gpu_layers", 999)),
            "--flash-attn",
            "on",
            "--cache-type-k",
            kv_k,
            "--cache-type-v",
            kv_v,
            "--ctx-size",
            str(config.get("context", 262144)),
            "--host",
            "0.0.0.0",
            "--port",
            str(config.get("port", 8081)),
            "--jinja",
        ]
