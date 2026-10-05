from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from appliance.db_ext import WorkbenchDB
from appliance.search.candidates import (
    generate_prune_candidates,
    generate_quant_candidates,
)
from appliance.search.pareto import frontier
from appliance.search.quant_recovery_loop import QuantRecoveryLoop
from appliance.recovery.escalation import RecoveryPolicy


@dataclass(frozen=True)
class ReleaseSearchConfig:
    source_candidate: str
    work_dir: Path
    max_candidates: int = 64
    require_262k: bool = True
    max_vram_mib: float = 11776.0
    oxcoder_wins_required: int = 6


class ReleaseSearch:
    """Controller for the opinionated '12 GB Release Search' workflow.

    Heavy stages are delegated through callbacks so the same controller can be
    used from the UI, API, or future queue worker.
    """

    def __init__(
        self,
        db: WorkbenchDB,
        *,
        launch_stage: Callable[[str, dict[str, Any]], str],
    ) -> None:
        self.db = db
        self.launch_stage = launch_stage
        self.quant_recovery = QuantRecoveryLoop(RecoveryPolicy(), launch_stage)

    def plan(self, config: ReleaseSearchConfig) -> dict[str, Any]:
        prune = generate_prune_candidates({})
        quant = generate_quant_candidates({})
        return {
            "source_candidate": config.source_candidate,
            "prune_candidates": prune[: config.max_candidates],
            "quant_templates": quant,
            "recovery_policy": {
                "ptq_accept_retention": 0.98,
                "qat_escalation_retention": 0.95,
                "max_recovery_rounds": 2,
                "requantize_after_recovery": True,
                "reevaluate_after_requantize": True,
            },
            "hard_gates": {
                "max_vram_mib": config.max_vram_mib,
                "context_tokens": 262144 if config.require_262k else None,
                "oxcoder_wins_required": config.oxcoder_wins_required,
            },
        }


    def recovery_action(
        self,
        *,
        candidate_metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
        recovery_round: int,
        stage_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Decide and optionally launch recovery for a quantized candidate.

        The caller must re-run the original quantization backend and evaluation
        after a recovery job completes before updating the Pareto frontier.
        """
        return self.quant_recovery.next_action(
            candidate_metrics=candidate_metrics,
            reference_metrics=reference_metrics,
            recovery_round=recovery_round,
            config=stage_config,
        )

    def pareto_frontier(self) -> list[dict[str, Any]]:
        return frontier(self.db.list_candidates(10000))

    def write_plan(self, config: ReleaseSearchConfig) -> Path:
        config.work_dir.mkdir(parents=True, exist_ok=True)
        path = config.work_dir / "release-search-plan.json"
        path.write_text(json.dumps(self.plan(config), indent=2), encoding="utf-8")
        return path
