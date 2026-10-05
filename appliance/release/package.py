from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_release(
    *,
    candidate: dict[str, Any],
    output_dir: Path,
    benchmark_report: dict[str, Any],
    provenance: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = Path(candidate["artifact_path"])
    manifest = {
        "candidate": candidate,
        "benchmark_report": benchmark_report,
        "provenance": provenance,
        "artifact": {
            "path": str(artifact),
            "sha256": sha256(artifact) if artifact.is_file() else None,
        },
    }
    (output_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    model_card = f"""---
library_name: llama.cpp
---

# Qwen3.6 12GB release candidate

## Hardware target
Single 12 GB GPU with a 262,144-token runtime context target.

## Compression
Backend: {candidate.get("runtime")}

## Benchmarks
```json
{json.dumps(benchmark_report, indent=2)}
```

## Provenance
```json
{json.dumps(provenance, indent=2)}
```

See `release-manifest.json` for exact machine-readable metadata.
"""
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")
    return output_dir
