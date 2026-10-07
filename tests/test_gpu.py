"""Cross-vendor discovery, telemetry and precision behavior."""
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from appliance import gpu
from appliance.core import NvidiaTelemetry, peak_vram


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


def amd_fixture(tmp_path, *, gpu_index=0, minor=129, visible=True):
    sysroot, devroot = tmp_path / 'sys', tmp_path / 'dev'
    put(devroot / 'kfd', '')
    put(sysroot / 'class/kfd/kfd/topology/nodes/0/gpu_id', '0')
    for index in range(gpu_index + 1):
        node = sysroot / f'class/kfd/kfd/topology/nodes/{index + 1}'
        put(node / 'gpu_id', 100 + index)
        put(node / 'name', 'navy_flounder')
        put(node / 'properties', f'drm_render_minor {minor + index}\n')
    if visible:
        put(devroot / f'dri/renderD{minor + gpu_index}', '')
    device = sysroot / f'class/drm/renderD{minor + gpu_index}/device'
    for name, value in {'vendor':'0x1002', 'mem_info_vram_total':12*1024**3,
                        'mem_info_vram_used':3*1024**3, 'gpu_busy_percent':42,
                        'hwmon/hwmon7/temp1_input':47000,
                        'hwmon/hwmon7/power1_average':99000000}.items():
        put(device / name, value)
    return sysroot, devroot, device


def test_nvidia_discovery_preserves_fields(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout='0, NVIDIA T1200, 4096, 500, 10, 41, 12.5\n'))
    monkeypatch.setenv('QWEN12G_GPU_BACKEND', 'cuda')
    row, = gpu.gpu_info()
    assert row['vendor'] == 'NVIDIA' and row['memory_used'] == '500'
    assert row['index'] == '0' and row['power'] == '12.5'


def test_amd_discovery_units_and_physical_index(tmp_path):
    sysroot, devroot, device = amd_fixture(tmp_path, gpu_index=1)
    row, = gpu.amd_info(sysroot, devroot)
    assert row['index'] == '1'  # inaccessible first GPU must not renumber the second
    assert float(row['memory_total']) == 12288
    assert float(row['memory_used']) == 3072
    assert float(row['temperature']) == 47
    assert float(row['power']) == 99
    (device / 'mem_info_vram_used').unlink()
    assert gpu.amd_info(sysroot, devroot)[0]['memory_used'] == 'N/A'


def test_amd_requires_exposed_devices(tmp_path):
    sysroot, devroot, _ = amd_fixture(tmp_path, visible=False)
    assert gpu.amd_info(sysroot, devroot) == []
    (devroot / 'kfd').unlink()
    assert gpu.amd_info(sysroot, devroot) == []


def test_auto_fallback_and_explicit_backend(monkeypatch):
    monkeypatch.setattr(gpu, 'nvidia_info', list)
    monkeypatch.setattr(gpu, 'amd_info', lambda: [{'vendor':'AMD'}])
    monkeypatch.delenv('QWEN12G_GPU_BACKEND', raising=False)
    assert gpu.gpu_info() == [{'vendor':'AMD'}]
    monkeypatch.setenv('QWEN12G_GPU_BACKEND', 'cuda')
    assert gpu.gpu_info() == []
    monkeypatch.setenv('QWEN12G_GPU_BACKEND', 'rocm')
    monkeypatch.setattr(gpu, 'nvidia_info', lambda: [{'vendor':'NVIDIA'}])
    assert gpu.gpu_info() == [{'vendor':'AMD'}]


def test_inventory_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 3
        raise subprocess.TimeoutExpired('nvidia-smi', 3)
    monkeypatch.setattr(subprocess, 'run', timeout)
    assert gpu.nvidia_info() == []


def test_amd_telemetry_flows_into_existing_peak_reader(tmp_path, monkeypatch):
    sysroot, devroot, _ = amd_fixture(tmp_path)
    recorder = gpu.GpuTelemetry(tmp_path / 'trace.csv')
    def sample():
        recorder.stop_event.set()
        return gpu.amd_info(sysroot, devroot)
    monkeypatch.setattr(gpu, 'gpu_info', sample)
    recorder._run()
    assert peak_vram(recorder.output) == {'0': 3072.0}
    assert NvidiaTelemetry is gpu.GpuTelemetry


def test_rocm_precision_and_kbit_guard(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, 'hip', '7.2')
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda **kw: False)
    assert gpu.compute_dtype() == torch.float32
    assert gpu.compute_dtype('cpu') == torch.float32
    with pytest.raises(ValueError, match='load_in_4bit=false'):
        gpu.require_kbit_support(True)
    gpu.require_kbit_support(False)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda **kw: True)
    assert gpu.compute_dtype() == torch.bfloat16
    monkeypatch.setattr(torch.version, 'hip', None)
    assert gpu.compute_dtype() == torch.bfloat16
    gpu.require_kbit_support(True)
