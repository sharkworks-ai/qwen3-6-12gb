from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich import print
from rich.table import Table

from qwen12g.db import RunDB
from qwen12g.jobs import (
    launch_stage,
    parse_result,
    read_remote_run_file,
    remote_storage,
    run_logs,
    stop_run,
    sync_run_status,
)
from qwen12g.system_info import full_environment, write_probe
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
run_app = typer.Typer(
    help="Launch registered experiment stages on the remote worker.",
    no_args_is_help=True,
)
runs_app = typer.Typer(
    help="Inspect and control recorded remote runs.",
    no_args_is_help=True,
)
system_app = typer.Typer(
    help="Capture control-plane and compute-plane environment details.",
    no_args_is_help=True,
)
app.add_typer(worker_app, name="worker")
app.add_typer(run_app, name="run")
app.add_typer(runs_app, name="runs")
app.add_typer(system_app, name="system")

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
    """Validate Docker connectivity, worker image, GPUs, and persistent storage."""
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
        print(
            f"[red]Expected {config.expected_gpus} GPUs but detected {len(gpus)}.[/red]"
        )
        raise typer.Exit(code=2)

    if config.expected_gpu_model:
        mismatches = [gpu for gpu in gpus if config.expected_gpu_model not in gpu]
        if mismatches:
            print(
                "[yellow]Warning: one or more GPUs do not match the configured "
                "model string.[/yellow]"
            )
            for gpu in mismatches:
                print(f"  {gpu}")

    storage = remote_storage(config)
    available_gib = storage["available_bytes"] / (1024**3)
    size_gib = storage["size_bytes"] / (1024**3)
    print(f"Worker storage: {available_gib:.1f} GiB free of {size_gib:.1f} GiB")

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


@worker_app.command("storage")
def worker_storage(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Show persistent worker storage capacity."""
    storage = remote_storage(load_worker_config(config_path))
    print(storage)


@worker_app.command("logs")
def worker_show_logs(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
    tail: Annotated[int, typer.Option("--tail", min=1, max=10000)] = 200,
) -> None:
    """Show recent trainer-service logs without opening a remote shell."""
    worker_logs(load_worker_config(config_path), tail=tail)


@run_app.command("smoke")
def run_smoke(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
    gpu_mode: Annotated[
        str,
        typer.Option("--gpu-mode", help="distributed, gpu0, or gpu1"),
    ] = "distributed",
) -> None:
    """Launch a detached GPU smoke run with telemetry and persistent results."""
    config = load_worker_config(config_path)
    record = launch_stage(config, stage="smoke", gpu_mode=gpu_mode)
    print(f"[green]Launched {record.run_id}[/green]")
    print(f"Container: {record.container_name}")
    print(f"Git commit: {record.git_commit}")
    print(f"Image ID: {record.image_id}")


@runs_app.command("list")
def runs_list(
    limit: Annotated[int, typer.Option("--limit", min=1, max=500)] = 50,
) -> None:
    """List runs recorded by the laptop control plane."""
    records = RunDB().list(limit=limit)
    table = Table("Run ID", "Stage", "Status", "GPU mode", "Created")
    for record in records:
        table.add_row(
            record.run_id,
            record.stage,
            record.status,
            record.gpu_mode,
            record.created_at,
        )
    print(table)


@runs_app.command("status")
def runs_status(
    run_id: str,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Reconnect to a detached run and synchronize its Docker state."""
    config = load_worker_config(config_path)
    record = sync_run_status(config, run_id)
    print(record)


@runs_app.command("logs")
def runs_logs(
    run_id: str,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
    tail: Annotated[int, typer.Option("--tail", min=1, max=10000)] = 200,
) -> None:
    """Show Docker logs for a detached run."""
    db = RunDB()
    record = db.get(run_id)
    if record is None:
        print(f"[red]Unknown run ID: {run_id}[/red]")
        raise typer.Exit(code=2)
    print(run_logs(load_worker_config(config_path), record, tail=tail))


@runs_app.command("result")
def runs_result(
    run_id: str,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Read a run's persistent worker-side result.json."""
    config = load_worker_config(config_path)
    payload = parse_result(read_remote_run_file(config, run_id, "result.json"))
    if payload is None:
        print("[yellow]result.json is not available yet.[/yellow]")
        raise typer.Exit(code=1)
    print(payload)


@runs_app.command("stop")
def runs_stop(
    run_id: str,
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
) -> None:
    """Stop a detached run without deleting its persistent outputs."""
    db = RunDB()
    record = db.get(run_id)
    if record is None:
        print(f"[red]Unknown run ID: {run_id}[/red]")
        raise typer.Exit(code=2)
    config = load_worker_config(config_path)
    stop_run(config, record)
    updated = sync_run_status(config, run_id, db=db)
    print(f"{run_id}: {updated.status}")


@system_app.command("probe")
def system_probe(
    config_path: Annotated[Path, typer.Option("--config")] = DEFAULT_WORKER_CONFIG,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Capture laptop and remote-worker environment metadata."""
    config = load_worker_config(config_path)
    payload = full_environment(config)
    if output is not None:
        write_probe(output, payload)
        print(f"[green]Wrote {output}[/green]")
    else:
        print(payload)


if __name__ == "__main__":
    app()
