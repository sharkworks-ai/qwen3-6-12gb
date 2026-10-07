from __future__ import annotations
import re
from dataclasses import dataclass
import torch
from torch.nn.utils import parametrize
from appliance.qat.fake_quant import FakeQuantSpec, fake_quant_weight


class FakeQuantParametrization(torch.nn.Module):
    def __init__(self, spec: FakeQuantSpec):
        super().__init__()
        self.spec = spec

    def forward(self, x):
        return fake_quant_weight(x, self.spec)


@dataclass(frozen=True)
class Rule:
    pattern: str
    spec: FakeQuantSpec


def rules_for(mode: str):
    if mode == "q3_moe":
        return [Rule(r"\.experts\.(gate_up_proj|down_proj)$", FakeQuantSpec(3, 64))]
    if mode == "q2_moe":
        return [Rule(r"\.experts\.(gate_up_proj|down_proj)$", FakeQuantSpec(2, 64))]
    if mode == "ternary_moe":
        return [Rule(r"\.experts\.(gate_up_proj|down_proj)$", FakeQuantSpec(1.58, 64, True))]
    if mode == "q4_dense":
        return [
            Rule(
                r"\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)\.weight$",
                FakeQuantSpec(4, 64),
            )
        ]
    raise ValueError(mode)


def apply_fake_quant(model: torch.nn.Module, mode: str) -> list[str]:
    matched = []
    module_map = dict(model.named_modules())
    for full_name, _ in list(model.named_parameters()):
        for rule in rules_for(mode):
            if not re.search(rule.pattern, full_name):
                continue
            module_name, param_name = full_name.rsplit(".", 1)
            module = module_map[module_name]
            if parametrize.is_parametrized(module, param_name):
                break
            parametrize.register_parametrization(
                module, param_name, FakeQuantParametrization(rule.spec), unsafe=True
            )
            matched.append(full_name)
            break
    return matched


def apply_precision_map(model: torch.nn.Module, precision: dict) -> list[str]:
    parameters = dict(model.named_parameters())
    modules = dict(model.named_modules())
    targets = precision["tensors"]
    missing = set(targets) - set(parameters)
    if missing or not targets:
        raise ValueError(f"Empty or stale precision map: {sorted(missing)[:8]}")
    for name, item in targets.items():
        parent, parameter = name.rsplit(".", 1)
        spec = FakeQuantSpec(item["bits"], item["group_size"], item["bits"] == 1.58)
        parametrize.register_parametrization(
            modules[parent],
            parameter,
            FakeQuantParametrization(spec),
            unsafe=True,
        )
    return list(targets)


def remove_fake_quant(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
