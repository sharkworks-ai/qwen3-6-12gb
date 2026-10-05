from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourcePolicy:
    max_concurrent_jobs: int = 2
    max_disk_gib: float = 1500.0
    max_host_ram_gib: float = 256.0
    max_gpu_temp_c: int = 88
    max_gpu_power_w: float | None = None
    reserve_disk_gib: float = 50.0


def can_launch(
    *,
    running_jobs: int,
    free_disk_gib: float,
    gpu_temps: list[float],
    policy: ResourcePolicy,
) -> tuple[bool, str]:
    if running_jobs >= policy.max_concurrent_jobs:
        return False, "concurrency limit reached"
    if free_disk_gib < policy.reserve_disk_gib:
        return False, "insufficient free disk"
    if gpu_temps and max(gpu_temps) >= policy.max_gpu_temp_c:
        return False, "GPU temperature limit reached"
    return True, "ok"
