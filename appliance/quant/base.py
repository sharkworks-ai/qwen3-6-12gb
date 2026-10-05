from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantContext:
    source_model: Path
    output_dir: Path
    calibration_file: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantResult:
    backend: str
    status: str
    artifacts: list[str]
    runtime: str
    notes: list[str] = field(default_factory=list)


class QuantBackend(ABC):
    name: str
    experimental: bool = False

    @abstractmethod
    def probe(self, context: QuantContext) -> dict[str, Any]:
        """Return compatibility/readiness information without mutating the model."""

    @abstractmethod
    def run(self, context: QuantContext) -> QuantResult:
        """Run quantization and return produced artifacts."""

    @abstractmethod
    def runtime_command(self, artifact: Path, config: dict[str, Any]) -> list[str]:
        """Return a recommended runtime command for the produced artifact."""
