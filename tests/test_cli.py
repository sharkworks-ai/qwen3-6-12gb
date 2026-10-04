from typer.testing import CliRunner

from qwen12g.cli import app


def test_doctor() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Qwen3.6 12GB lab" in result.stdout
