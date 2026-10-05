from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalContext:
    model: Path
    output_dir: Path
    runtime: str
    config: dict[str, Any]


class EvalAdapter(ABC):
    name: str

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def run(self, context: EvalContext) -> dict[str, Any]: ...
