from pathlib import Path

from appliance.datasets.contamination import overlap_score
from appliance.runtime.scheduler import choose_allocation
from appliance.search.expert_importance import combine_importance
from appliance.search.pareto import dominates, frontier
from appliance.search.sensitivity import assign_precision
from appliance.eval.oxcoder_gate import score


def test_pareto():
    a = {"candidate_id": "a", "metrics": {"agent_score": 10, "peak_vram_mib": 10}}
    b = {"candidate_id": "b", "metrics": {"agent_score": 9, "peak_vram_mib": 11}}
    assert dominates(a["metrics"], b["metrics"])
    assert frontier([a, b]) == [a]


def test_oxcoder_gate_counts():
    results = {
        "swe_bench_verified": 74,
        "terminal_bench_terminus2": 51,
        "nl2repo": 40,
        "mcp_atlas": 60,
        "browsecomp": 60,
        "claweval": 70,
        "gpqa_diamond": 87,
    }
    scored = score(results)
    assert scored["overall_wins"] == 7
    assert scored["swe_guardrail"] is True
    assert scored["terminal_guardrail"] is True


def test_scheduler():
    allocation = choose_allocation(
        requested_mode="independent", gpu_count=2, busy_devices={0}
    )
    assert allocation.devices == (1,)


def test_contamination_overlap():
    assert overlap_score(
        "alpha beta gamma delta epsilon zeta eta theta iota",
        "alpha beta gamma delta epsilon zeta eta theta iota",
    ) == 1.0


def test_precision_assignment():
    result = assign_precision({"a": 0.9, "b": 0.5, "c": 0.1})
    assert result == {"a": "Q5_K", "b": "Q4_K", "c": "Q3_K"}


def test_importance():
    value = combine_importance(
        frequency=1,
        routing_mass=1,
        activation_norm=1,
        ablation_drop=1,
        task_scores={"coding": 1, "agent": 1, "long_context": 1},
    )
    assert value > 1
