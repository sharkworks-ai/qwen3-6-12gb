"""Pinned upstream components loaded without colliding with the project's src."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

GSQ_REVISION = "03fc16484c369e3127225615d5e03e8d3a6043e3"
CATQ_REVISION = "5a8fcd4f7e0366554b732d300a37e6ea467c3c35"


def check_checkout(root: Path, revision: str) -> dict:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"Upstream revision mismatch at {root}: expected {revision}, got {head}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
    return {"root": str(root), "revision": head}


def load_components(cfg: dict):
    gsq = Path(cfg.get("gsq_root", "/opt/GSQ"))
    catq = Path(cfg.get("bittern_root", "/opt/BitTern"))
    provenance = {
        "gsq": check_checkout(gsq, GSQ_REVISION),
        "packages": {name: version(name) for name in ("torch", "transformers", "lion-pytorch")},
    }
    package = "_qwen12g_pinned_gsq"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__path__ = [str(gsq / "src")]
        sys.modules[package] = module
    from importlib import import_module

    prior = import_module(package + ".prior.gptq")
    quant = import_module(package + ".quantization")
    ternary = None
    if cfg["preset"] == "extreme":
        provenance["catq"] = check_checkout(catq, CATQ_REVISION)
        path = catq / "projects/cat-q/quantize/quantizer.py"
        spec = importlib.util.spec_from_file_location("_qwen12g_pinned_catq", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ternary = module.TernaryQuantizer
    return prior, quant, ternary, provenance
