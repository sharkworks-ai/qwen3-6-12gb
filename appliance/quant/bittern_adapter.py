from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def find_candidate_entrypoint(project: Path, configured: str | None) -> Path:
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = project / path
        if path.exists():
            return path
        raise FileNotFoundError(path)

    # Upstream CAT-Q layout is still evolving. Discover a plausible official
    # quantization entrypoint and refuse ambiguity rather than silently guessing.
    candidates = []
    for pattern in ("*quant*.py", "**/*quant*.py", "*main*.py", "**/*main*.py"):
        candidates.extend(project.glob(pattern))
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Unable to uniquely determine the installed CAT-Q entrypoint. "
        "Set catq_entrypoint in the BitTern backend config. Candidates: "
        + ", ".join(str(p) for p in candidates[:20])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bittern-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--calibration")
    args = parser.parse_args()

    cfg = json.loads(args.config_json)
    project = Path(args.bittern_root) / "projects/cat-q"
    entrypoint = find_candidate_entrypoint(project, cfg.get("catq_entrypoint"))

    # Command mapping is deliberately explicit/configurable because CAT-Q's
    # public CLI is not yet a stable Qwen3.6 contract.
    template = cfg.get("catq_args")
    if not template:
        raise RuntimeError(
            "BitTern is installed but no stable Qwen3.6 CAT-Q CLI contract is "
            "available. Set catq_entrypoint and catq_args for the pinned BitTern "
            "revision after running the compatibility self-test."
        )

    substitutions = {
        "{model}": args.model,
        "{output}": args.output,
        "{calibration}": args.calibration or "",
    }
    mapped = []
    for value in template:
        for key, replacement in substitutions.items():
            value = value.replace(key, replacement)
        if value:
            mapped.append(value)

    subprocess.run(["python3", str(entrypoint), *mapped], check=True)


if __name__ == "__main__":
    main()
