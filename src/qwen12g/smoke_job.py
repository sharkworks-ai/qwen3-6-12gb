from __future__ import annotations

import json
import time

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPUs are visible")

    allocations = []
    for index in range(gpu_count):
        with torch.cuda.device(index):
            tensor = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=index)
            tensor.fill_(index + 1)
            allocations.append(tensor)

    time.sleep(3)

    print(
        json.dumps(
            {
                "cuda": True,
                "gpu_count": gpu_count,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
            },
            sort_keys=True,
        )
    )

    del allocations
    for index in range(gpu_count):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
