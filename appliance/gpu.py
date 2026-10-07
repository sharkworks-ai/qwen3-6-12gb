"""GPU inventory and telemetry without importing a training runtime into the web app.

ROCm keeps PyTorch's cuda device names. Physical indices here follow KFD GPU
node order; invisible render nodes keep their index so selection is not renumbered.
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

FIELDS = ("index", "name", "memory_total", "memory_used", "utilization", "temperature", "power")


def _read(path: Path, default="N/A"):
    try:
        return path.read_text().strip()
    except OSError:
        return default


def _scaled(path: Path, divisor):
    try:
        return str(float(path.read_text().strip()) / divisor)
    except (OSError, ValueError):
        return "N/A"


def nvidia_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for values in csv.reader(result.stdout.splitlines()):
        if len(values) == len(FIELDS):
            rows.append({**dict(zip(FIELDS, (v.strip() for v in values))), "vendor": "NVIDIA"})
    return rows


def amd_info(sys_root=Path("/sys"), dev_root=Path("/dev")):
    if not (dev_root / "kfd").exists():
        return []
    topology = sys_root / "class/kfd/kfd/topology/nodes"
    nodes = sorted((p for p in topology.glob("*") if p.name.isdigit()), key=lambda p: int(p.name))
    rows, index = [], 0
    for node in nodes:
        gpu_id = _read(node / "gpu_id", "0")
        if not gpu_id.isdigit() or int(gpu_id) == 0:
            continue
        physical_index = index
        index += 1
        props = dict(line.split(maxsplit=1) for line in _read(node / "properties", "").splitlines() if " " in line)
        minor = props.get("drm_render_minor", "")
        if not minor.isdigit() or not (dev_root / "dri" / f"renderD{minor}").exists():
            continue
        device = sys_root / "class/drm" / f"renderD{minor}" / "device"
        if _read(device / "vendor", "") != "0x1002":
            continue
        hwmon = next(iter(sorted((device / "hwmon").glob("hwmon*"))), device / "missing")
        rows.append({
            "index": str(physical_index), "vendor": "AMD",
            "name": "AMD " + _read(device / "product_name", _read(node / "name", "GPU")),
            "memory_total": _scaled(device / "mem_info_vram_total", 1024**2),
            "memory_used": _scaled(device / "mem_info_vram_used", 1024**2),
            "utilization": _read(device / "gpu_busy_percent"),
            "temperature": _scaled(hwmon / "temp1_input", 1000),
            "power": _scaled(hwmon / "power1_average", 1000000),
        })
    return rows


def gpu_info():
    backend = os.environ.get("QWEN12G_GPU_BACKEND", "auto")
    if backend not in {"auto", "cuda", "rocm"}:
        raise ValueError("QWEN12G_GPU_BACKEND must be auto, cuda or rocm")
    if backend != "rocm":
        rows = nvidia_info()
        if rows or backend == "cuda":
            return rows
    return amd_info()


class GpuTelemetry:
    def __init__(self, output: Path, interval: float = 1.0):
        self.output, self.interval = output, interval
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "index", "name", "memory_total_mib", "memory_used_mib",
                             "utilization_gpu_percent", "temperature_c", "power_w"])
            while not self.stop_event.is_set():
                now = datetime.now(UTC).isoformat()
                rows = gpu_info()
                for row in rows:
                    writer.writerow([now, *(row.get(k, "N/A") for k in FIELDS)])
                if not rows:
                    writer.writerow([now, "error", "No GPU telemetry available"])
                stream.flush()
                self.stop_event.wait(self.interval)


def compute_dtype(device="cuda:0"):
    """Keep NVIDIA behavior; avoid unsupported native BF16 on older Radeon GPUs."""
    import torch
    if str(device) == "cpu":
        return torch.float32
    if torch.version.hip:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                return torch.float32
    return torch.bfloat16


def require_kbit_support(enabled):
    import torch
    if enabled and torch.version.hip:
        raise ValueError("4-bit bitsandbytes loading is not enabled in the ROCm image. "
                         "Set load_in_4bit=false and size the model for full-precision loading.")
