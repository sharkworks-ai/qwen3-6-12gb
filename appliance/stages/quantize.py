from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from appliance.stages.common import load_config, save_json


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    llama_root = Path(config.get("llama_cpp_root", "/opt/llama.cpp"))
    convert = llama_root / "convert_hf_to_gguf.py"
    quantize = llama_root / "build/bin/llama-quantize"
    imatrix_bin = llama_root / "build/bin/llama-imatrix"
    for binary in (convert, quantize):
        if not binary.exists():
            raise FileNotFoundError(binary)

    model = Path(config["model"]).resolve()
    if not model.exists():
        raise FileNotFoundError("Quantization currently expects a local HF checkpoint directory")

    output_dir = Path(config.get("output_dir", "/data/artifacts/quantized"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_gguf = output_dir / config.get("base_gguf_name", "model-BF16.gguf")
    final_gguf = output_dir / config.get("output_name", "model-Q3_K_M.gguf")
    log = output_dir / "quantize.log"

    commands: list[list[str]] = []
    if not base_gguf.exists() or bool(config.get("reconvert", False)):
        commands.append([
            os.environ.get("PYTHON", "python3"), str(convert), str(model),
            "--outfile", str(base_gguf), "--outtype", config.get("convert_outtype", "bf16"),
        ])

    imatrix_path = None
    calibration = config.get("calibration_file")
    if calibration:
        if not imatrix_bin.exists():
            raise FileNotFoundError(imatrix_bin)
        imatrix_path = output_dir / "imatrix.gguf"
        commands.append([
            str(imatrix_bin), "-m", str(base_gguf), "-f", str(calibration),
            "-ngl", str(config.get("imatrix_gpu_layers", 999)),
            "-c", str(config.get("imatrix_context", 4096)),
            "--no-ppl", "-o", str(imatrix_path),
        ])

    qcmd = [str(quantize)]
    if imatrix_path is not None:
        qcmd += ["--imatrix", str(imatrix_path)]
    if config.get("output_tensor_type"):
        qcmd += ["--output-tensor-type", config["output_tensor_type"]]
    if config.get("token_embedding_type"):
        qcmd += ["--token-embedding-type", config["token_embedding_type"]]
    for override in config.get("tensor_overrides", []):
        # override form: {"pattern": "regex", "type": "q5_k_m"}
        qcmd += ["--tensor-type", f"{override['pattern']}={override['type']}"]
    qcmd += [str(base_gguf), str(final_gguf), config.get("quant_type", "q3_k_m")]
    commands.append(qcmd)

    if config.get("dry_run"):
        save_json(output_dir / "quantize_plan.json", {"commands": commands})
        print(json.dumps({"dry_run": True, "commands": commands}, indent=2))
        return

    for command in commands:
        run(command, log)

    commit_file = llama_root / ".git" / "HEAD"
    save_json(output_dir / "quantization_manifest.json", {
        "model": str(model),
        "output": str(final_gguf),
        "quant_type": config.get("quant_type", "q3_k_m"),
        "imatrix": str(imatrix_path) if imatrix_path else None,
        "tensor_overrides": config.get("tensor_overrides", []),
    })
    print(json.dumps({"output": str(final_gguf)}))


if __name__ == "__main__":
    main()
