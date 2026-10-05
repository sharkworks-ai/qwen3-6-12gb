from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricSet:
    coding_score: float | None = None
    agent_score: float | None = None
    long_context_score: float | None = None
    oxcode_wins: int | None = None
    peak_vram_mib: float | None = None
    decode_tps: float | None = None
    prefill_tps: float | None = None
    artifact_size_bytes: int | None = None
    context_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_id: str | None
    stage: str
    config: dict[str, Any]
    metrics: MetricSet
    artifact_path: str | None = None
    runtime: str | None = None
    status: str = "created"
