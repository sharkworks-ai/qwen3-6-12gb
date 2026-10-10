from __future__ import annotations
import re
from dataclasses import dataclass
import torch
from torch.nn.utils import parametrize
from appliance.qat.fake_quant import FakeQuantSpec, fake_quant_on_grid, fake_quant_weight


class FakeQuantParametrization(torch.nn.Module):
    """Fake quantization onto a reconstruction's fixed grid when given, else min/max."""

    def __init__(self, spec: FakeQuantSpec, grid: torch.Tensor | None = None):
        super().__init__()
        self.spec = spec
        self.register_buffer("grid", grid, persistent=False)

    def quantize(self, x):
        if self.grid is None:
            return fake_quant_weight(x, self.spec)
        return fake_quant_on_grid(x, self.spec, self.grid)

    def forward(self, x):
        return self.quantize(x)


class LowRankFakeQuantParametrization(FakeQuantParametrization):
    """fake_quant(W + B @ A): a trainable low-rank correction seen through the quantizer.

    W stays frozen, so recovery needs memory for the model plus small adapters rather
    than full-parameter optimizer state. Removing the parametrization leaves the merged,
    fake-quantized weight that re-quantization starts from. Batched (expert) weights get
    one adapter per leading index.
    """

    def __init__(
        self,
        spec: FakeQuantSpec,
        weight: torch.Tensor,
        rank: int,
        grid: torch.Tensor | None = None,
    ):
        super().__init__(spec, grid)
        *batch, rows, columns = weight.shape
        options = {"device": weight.device, "dtype": torch.float32}
        # B starts at zero, so training begins from the plain fake-quantized weight.
        self.lora_b = torch.nn.Parameter(torch.zeros(*batch, rows, rank, **options))
        self.lora_a = torch.nn.Parameter(
            torch.randn(*batch, rank, columns, **options) / columns**0.5
        )

    def forward(self, x):
        delta = torch.matmul(self.lora_b, self.lora_a).to(x.dtype)
        return self.quantize(x + delta)


def _parametrization(spec, module, param_name, rank, grid=None):
    weight = getattr(module, param_name)
    # Packed groups pad each row; flattened fake-quant groups only line up without padding.
    if grid is not None and weight.shape[-1] % spec.group_size:
        grid = None
    if grid is not None:
        grid = grid.to(device=weight.device, dtype=torch.float32)
    if rank:
        return LowRankFakeQuantParametrization(spec, weight, rank, grid)
    return FakeQuantParametrization(spec, grid)


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


def apply_fake_quant(model: torch.nn.Module, mode: str, rank: int = 0) -> list[str]:
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
                module,
                param_name,
                _parametrization(rule.spec, module, param_name, rank),
                unsafe=True,
            )
            matched.append(full_name)
            break
    return matched


def apply_precision_map(
    model: torch.nn.Module,
    precision: dict,
    rank: int = 0,
    grids: dict[str, torch.Tensor] | None = None,
) -> list[str]:
    """Register fake quantization for every precision-map tensor.

    `grids` maps tensor names to the per-group scales a reconstruction packed; those
    tensors train on exactly that grid, so the starting point is the reconstruction.
    """
    parameters = dict(model.named_parameters())
    modules = dict(model.named_modules())
    targets = precision["tensors"]
    missing = set(targets) - set(parameters)
    if missing or not targets:
        raise ValueError(f"Empty or stale precision map: {sorted(missing)[:8]}")
    grids = grids or {}
    for name, item in targets.items():
        parent, parameter = name.rsplit(".", 1)
        spec = FakeQuantSpec(item["bits"], item["group_size"], item["bits"] == 1.58)
        parametrize.register_parametrization(
            modules[parent],
            parameter,
            _parametrization(spec, modules[parent], parameter, rank, grids.get(name)),
            unsafe=True,
        )
    return list(targets)


def load_reconstruction_grids(bundle) -> dict[str, torch.Tensor]:
    """Per-group scales of each packed tensor in a reconstruction bundle, by name."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    bundle = Path(bundle)
    manifest = bundle / "mixed-manifest.json"
    if not manifest.is_file():
        return {}
    tensors = json.loads(manifest.read_text()).get("tensors", {})
    return {name: load_file(str(bundle / item["file"]))["scales"] for name, item in tensors.items()}


def remove_fake_quant(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
