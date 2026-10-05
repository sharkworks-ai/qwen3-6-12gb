from __future__ import annotations

import json
from pathlib import Path

from .base import QuantBackend, QuantContext, QuantResult
from .utils import load_hf_config, run


class BitTernBackend(QuantBackend):
    """CAT-Q / BitTern ternary PTQ backend.

    Qwen3.6 is not currently listed among BitTern's officially released model
    targets, so Qwen3.6 requires explicit experimental opt-in.
    """

    name = "bittern_catq"
    experimental = True

    def probe(self, context: QuantContext) -> dict:
        cfg = load_hf_config(context.source_model)
        model_type = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")
        root = Path(context.config.get("bittern_root", "/opt/BitTern"))
        project = root / "projects/cat-q"

        officially_known = model_type in {
            "qwen3",
            "qwen3_moe",
        }
        qwen36_like = model_type in {
            "qwen3_5_moe",
            "qwen3_6_moe",
            "qwen3_5",
        }

        return {
            "compatible": project.exists() and (officially_known or qwen36_like),
            "officially_supported_architecture": officially_known,
            "experimental_qwen36_adapter_required": qwen36_like and not officially_known,
            "model_type": model_type,
            "bittern_root": str(root),
            "catq_project": str(project),
        }

    def run(self, context: QuantContext) -> QuantResult:
        probe = self.probe(context)
        cfg = context.config

        if not probe["compatible"]:
            raise RuntimeError(
                f"BitTern/CAT-Q compatibility probe failed for {probe['model_type']!r}"
            )

        if probe["experimental_qwen36_adapter_required"] and not cfg.get(
            "allow_experimental_qwen36", False
        ):
            raise RuntimeError(
                "Qwen3.6 is not an officially published BitTern target. "
                "Set allow_experimental_qwen36=true only after the appliance "
                "adapter self-test passes on the installed BitTern revision."
            )

        root = Path(cfg.get("bittern_root", "/opt/BitTern"))
        project = root / "projects/cat-q"
        context.output_dir.mkdir(parents=True, exist_ok=True)

        # BitTern evolves quickly. Rather than hard-code one unstable script name,
        # the image ships a stable adapter entrypoint which maps appliance config
        # onto the installed CAT-Q revision.
        adapter = Path(
            cfg.get(
                "adapter_script",
                "/app/appliance/quant/bittern_adapter.py",
            )
        )
        args = [
            "python3",
            str(adapter),
            "--bittern-root",
            str(root),
            "--model",
            str(context.source_model),
            "--output",
            str(context.output_dir),
            "--config-json",
            json.dumps(cfg),
        ]
        if context.calibration_file is not None:
            args += ["--calibration", str(context.calibration_file)]

        run(args)

        artifacts = [str(p) for p in context.output_dir.iterdir() if p.is_file()]
        return QuantResult(
            backend=self.name,
            status="succeeded",
            artifacts=artifacts,
            runtime="BitTern packed ternary runtime / CAT-Q inference",
            notes=[
                "1.58-bit ternary PTQ research backend.",
                "Qwen3.6 path is experimental until upstream BitTern explicitly supports it.",
                "Agent-loop benchmarks are mandatory before promotion.",
            ],
        )

    def runtime_command(self, artifact: Path, config: dict) -> list[str]:
        root = Path(config.get("bittern_root", "/opt/BitTern"))
        runner = config.get(
            "runtime_entrypoint",
            str(root / "projects/cat-q/deploy/run.py"),
        )
        return [
            "python3",
            runner,
            "--model",
            str(artifact),
        ]
