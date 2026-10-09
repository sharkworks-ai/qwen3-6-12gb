import subprocess
from pathlib import Path

from qwen12g import worker
from qwen12g.worker import compose_args, docker_args, load_worker_config


def test_load_worker_config() -> None:
    config = load_worker_config(Path("configs/worker/dual5090.yaml"))
    assert config.docker_context == "qwen5090"
    assert config.expected_gpus == 2
    assert config.service == "trainer"


def test_docker_args() -> None:
    config = load_worker_config(Path("configs/worker/dual5090.yaml"))
    assert docker_args(config, "info") == [
        "docker",
        "--context",
        "qwen5090",
        "info",
    ]


def test_compose_args() -> None:
    config = load_worker_config(Path("configs/worker/dual5090.yaml"))
    args = compose_args(config, "build", "trainer")
    assert args[:3] == ["docker", "--context", "qwen5090"]
    assert "docker/compose.worker.yml" in args


def test_gpu_inventory_bypasses_image_entrypoint(monkeypatch) -> None:
    config = load_worker_config(Path("configs/worker/dual5090.yaml"))
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(
            args, 0, "0, NVIDIA GeForce RTX 5090, 32607\n1, NVIDIA GeForce RTX 5090, 32607\n"
        )

    monkeypatch.setattr(worker, "run_command", fake_run)
    assert len(worker.gpu_inventory(config)) == 2
    args = seen["args"]
    assert args[args.index("--entrypoint") + 1] == "nvidia-smi"
    assert args.index("--entrypoint") < args.index(config.service)
