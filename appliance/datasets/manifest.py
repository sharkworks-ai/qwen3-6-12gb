from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    output: Path,
    *,
    name: str,
    sources: list[dict[str, Any]],
    licenses: list[dict[str, Any]],
    stats: dict[str, Any],
    files: list[Path],
) -> dict[str, Any]:
    payload = {
        "name": name,
        "sources": sources,
        "licenses": licenses,
        "stats": stats,
        "files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in files
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["manifest_sha256"] = sha256_file(output)
    return payload
