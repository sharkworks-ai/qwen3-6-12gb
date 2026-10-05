from __future__ import annotations

from .base import QuantBackend
from .bittern import BitTernBackend
from .llama_cpp import LlamaCppBackend
from .turboquant import TurboQuantBackend


BACKENDS: dict[str, QuantBackend] = {
    "llama_cpp": LlamaCppBackend(),
    "turboquant": TurboQuantBackend(),
    "bittern_catq": BitTernBackend(),
}


def get_backend(name: str) -> QuantBackend:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quantization backend {name!r}; choose from {sorted(BACKENDS)}"
        ) from exc
