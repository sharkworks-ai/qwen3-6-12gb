import json
import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

allocations = []
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        tensor = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device=index)
        tensor.fill_(index + 1)
        allocations.append(tensor)

print(json.dumps({
    "cuda": True,
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}), flush=True)
time.sleep(4)
