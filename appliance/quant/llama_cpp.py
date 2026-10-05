from __future__ import annotations

from pathlib import Path

from .base import QuantBackend, QuantContext, QuantResult
from .utils import run


class LlamaCppBackend(QuantBackend):
    name = "llama_cpp"

    def probe(self, context: QuantContext) -> dict:
        root = Path(context.config.get("llama_cpp_root", "/opt/llama.cpp"))
        quant = root / "build/bin/llama-quantize"
        convert = root / "convert_hf_to_gguf.py"
        return {
            "compatible": quant.exists() and convert.exists(),
            "llama_cpp_root": str(root),
            "quantizer": str(quant),
            "converter": str(convert),
        }

    def run(self, context: QuantContext) -> QuantResult:
        cfg = context.config
        root = Path(cfg.get("llama_cpp_root", "/opt/llama.cpp"))
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

        imatrix: Path | None = None
        if cfg.get("use_imatrix", True):
            if context.calibration_file is None:
                raise ValueError("llama.cpp imatrix quantization requires calibration_file")
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

        quant_type = cfg.get("quant_type", "Q3_K_M")
        output = context.output_dir / f"model-{quant_type}.gguf"
        args = [str(root / "build/bin/llama-quantize")]
        if imatrix is not None:
            args += ["--imatrix", str(imatrix)]

        for override in cfg.get("tensor_types", []):
            args += ["--tensor-type", override]

        args += [str(f16), str(output), quant_type]
        run(args)

        return QuantResult(
            backend=self.name,
            status="succeeded",
            artifacts=[str(output)],
            runtime="upstream llama.cpp",
            notes=["Supports per-tensor mixed precision through tensor_types overrides."],
        )

    def runtime_command(self, artifact: Path, config: dict) -> list[str]:
        root = Path(config.get("llama_cpp_root", "/opt/llama.cpp"))
        return [
            str(root / "build/bin/llama-server"),
            "-m",
            str(artifact),
            "-ngl",
            str(config.get("gpu_layers", 999)),
            "-c",
            str(config.get("context", 262144)),
            "--cache-type-k",
            config.get("cache_k", "q4_0"),
            "--cache-type-v",
            config.get("cache_v", "q4_0"),
            "--jinja",
            "--host",
            "0.0.0.0",
            "--port",
            str(config.get("port", 8081)),
        ]
