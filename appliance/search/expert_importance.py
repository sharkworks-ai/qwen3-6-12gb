from __future__ import annotations


def combine_importance(
    *,
    frequency: float,
    routing_mass: float,
    activation_norm: float,
    ablation_drop: float,
    coding_weight: float = 1.0,
    agent_weight: float = 1.0,
    long_context_weight: float = 1.0,
    task_scores: dict[str, float] | None = None,
) -> float:
    tasks = task_scores or {}
    task_component = (
        coding_weight * tasks.get("coding", 0.0)
        + agent_weight * tasks.get("agent", 0.0)
        + long_context_weight * tasks.get("long_context", 0.0)
    )
    return (
        0.20 * frequency
        + 0.25 * routing_mass
        + 0.15 * activation_norm
        + 0.25 * ablation_drop
        + 0.15 * task_component
    )
