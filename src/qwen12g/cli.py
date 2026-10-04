from pathlib import Path

import typer
import yaml
from rich import print

app = typer.Typer(help="Qwen3.6 12GB experiment controller")


@app.command()
def doctor() -> None:
    """Show the expected project state before a GPU worker is used."""
    print("[bold]Qwen3.6 12GB lab[/bold]")
    print("Repository bootstrap is installed.")
    print("Next: implement hardware probing, baseline runner, and evaluation harness.")


@app.command()
def show_config(path: Path = Path("configs/search/default.yaml")) -> None:
    """Load and print an experiment-search config."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(config)


if __name__ == "__main__":
    app()
