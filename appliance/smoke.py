"""Exercise the actual CUDA/HIP kernels, not just device enumeration."""
import json
import time
import torch
from appliance.gpu import compute_dtype, gpu_info

if not torch.cuda.is_available():
    raise RuntimeError("No usable NVIDIA CUDA or AMD ROCm GPU")

results = []
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        dtype = compute_dtype(f"cuda:{index}")
        x = torch.ones((256, 256), device=f"cuda:{index}", dtype=dtype, requires_grad=True)
        loss = (x @ x).float().mean()
        loss.backward()
        torch.cuda.synchronize(index)
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            raise RuntimeError("GPU arithmetic produced non-finite values")
        results.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "dtype": str(dtype), "matmul_backward": True})
print(json.dumps({"backend": "rocm" if torch.version.hip else "cuda",
                  "gpu_count": len(results), "devices": results,
                  "telemetry": gpu_info()}), flush=True)
time.sleep(4)
