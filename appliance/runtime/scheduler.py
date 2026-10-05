from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Allocation:
    mode: str
    devices: tuple[int, ...]


def choose_allocation(
    *,
    requested_mode: str,
    gpu_count: int,
    busy_devices: set[int] | None = None,
) -> Allocation:
    busy = busy_devices or set()
    free = [i for i in range(gpu_count) if i not in busy]

    if requested_mode == "distributed":
        if len(free) != gpu_count:
            raise RuntimeError("distributed job requires every GPU free")
        return Allocation("distributed", tuple(free))

    if requested_mode == "independent":
        if not free:
            raise RuntimeError("no GPU available")
        return Allocation("independent", (free[0],))

    if requested_mode.startswith("gpu"):
        index = int(requested_mode.removeprefix("gpu"))
        if index >= gpu_count or index in busy:
            raise RuntimeError(f"GPU {index} is unavailable")
        return Allocation(requested_mode, (index,))

    raise ValueError(f"unknown GPU scheduling mode: {requested_mode}")
