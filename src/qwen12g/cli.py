from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich import print

from qwen12g.worker import (
    build_worker,
    compose_config,
    create_context,
    gpu_inventory,
    inspect_context,
    load_worker_config,
    remote_info,
    start_worker,
    stop_worker,
    worker_logs,
)

app = typer.Typer(help="Qwen3.6 12GB experiment controller")
worker_app = typer.Typer(
    help="Control the remote Docker GPU worker from the laptop.",
    no_args_is_help=True,
)
app.add_typer(worker_app, name="worker")

DEFAULT_WORKER_CONFIG = Path("configs/worker/dual5090.yaml")


@app.command()
def doctor() -> None:
    """Show the expected project state before a GPU worker is used."""
    print("[bold]Qwen3.6 12GB lab[/bold]")
    print("Control plane: laptop")
    print("Compute plane: remote Docker GPU worker")
    print("Run [bold]qwen12g worker doctor[/bold] to validate the worker.")


@app.command()
def show_config(path: Path = Path("configs/search/default.yaml")) -> None:
    """Load and print an experiment-search config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(config)


@worker_app.command("context-create")
def worker_context_create(
    ssh_host: Annotated[
        str,
        typer.Argument(
            help="SSH target in user@host form, for example qwen-worker@training-host."
        ),
    ],
    context: Annotated[str, typer.Option("--context")] = "qwen5090",
) -> None:
    """Create the SSH-backed Docker context on the laptop."""
    create_context(context, ssh_host)
    print(f"[green]Created Docker context {context}[/green]")


@worker_app.command("doctor")
def worker_doctor(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Validate laptop-to-worker Docker connectivity and GPU inventory."""
    config = load_worker_config(config_path)

    inspect_context(config)
    remote_info(config)
    compose_config(config)

    print(f"[green]Docker context {config.docker_context} is reachable.[/green]")
    print("Building the worker image if needed...")
    build_worker(config)

    gpus = gpu_inventory(config)
    print(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  {gpu}")

    if len(gpus) != config.expected_gpus:
        raise typer.Exit(
            code=2,
        )

    if config.expected_gpu_model:
        mismatches = [
            gpu for gpu in gpus if config.expected_gpu_model not in gpu
        ]
        if mismatches:
            print(
                "[yellow]Warning: one or more GPUs do not match the configured "
                "model string.[/yellow]"
            )
            for gpu in mismatches:
                print(f"  {gpu}")

    print("[green]Remote worker passed the preliminary doctor checks.[/green]")


@worker_app.command("build")
def worker_build(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Build the worker image on the remote Docker daemon."""
    build_worker(load_worker_config(config_path))


@worker_app.command("start")
def worker_start(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Start the persistent trainer service on the remote worker."""
    start_worker(load_worker_config(config_path))


@worker_app.command("stop")
def worker_stop(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Stop project services on the remote worker."""
    stop_worker(load_worker_config(config_path))


@worker_app.command("gpus")
def worker_gpus(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Show GPUs visible inside the worker container."""
    config = load_worker_config(config_path)
    for gpu in gpu_inventory(config):
        print(gpu)


@worker_app.command("logs")
def worker_show_logs(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
    tail: Annotated[int, typer.Option("--tail", min=1, max=10000)] = 200,
) -> None:
    """Show recent trainer-service logs without opening a remote shell."""
    worker_logs(load_worker_config(config_path), tail=tail)


if __name__ == "__main__":
    app()
