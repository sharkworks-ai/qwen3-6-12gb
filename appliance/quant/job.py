from __future__ import annotations

import argparse
import json
from pathlib import Path

from .base import QuantContext
from .registry import get_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    backend = get_backend(cfg["backend"])
    context = QuantContext(
        source_model=Path(cfg["source_model"]),
        output_dir=Path(cfg["output_dir"]),
        calibration_file=(
            Path(cfg["calibration_file"])
            if cfg.get("calibration_file")
            else None
        ),
        config=cfg.get("backend_config", {}),
    )

    probe = backend.probe(context)
    print(json.dumps({"backend": backend.name, "probe": probe}, indent=2), flush=True)
    if not probe.get("compatible", False):
        raise RuntimeError(f"Backend probe failed: {probe}")

    result = backend.run(context)
    runtime = None
    if result.artifacts:
        runtime = backend.runtime_command(
            Path(result.artifacts[0]),
            context.config,
        )

    manifest = {
        "backend": result.backend,
        "status": result.status,
        "artifacts": result.artifacts,
        "runtime": result.runtime,
        "runtime_command": runtime,
        "notes": result.notes,
        "probe": probe,
    }
    context.output_dir.mkdir(parents=True, exist_ok=True)
    (context.output_dir / "quantization-result.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
