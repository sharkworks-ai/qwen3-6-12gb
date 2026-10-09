from pathlib import Path

from appliance.registry import command_for, get_job
from appliance.stages.prune import build_keep_map


def test_sft_uses_torchrun_for_two_processes(tmp_path: Path):
    cmd=command_for("sft",tmp_path/"config.json",{"num_processes":2})
    assert cmd[0]=="torchrun"
    assert "--nproc-per-node=2" in cmd


def test_prune_keep_map_selects_expected_experts():
    profile={"layers":{"0":{"count":[10,1,8,0],"routing_mass":[5,1,7,0]}}}
    keep=build_keep_map(profile,2,0.5)
    assert keep[0]==[0,2]


def test_registry_contains_pipeline():
    for name in ("sft","merge","profile","prune","calibration","quantize"):
        assert get_job(name).name==name
