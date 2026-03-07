from pathlib import Path

from click.testing import CliRunner

from auditml.cli import auditml


def test_cli_report_command(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "sample.csv").write_text(
        "attack,dataset,accuracy\n"
        "mia-threshold,mnist,0.6\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.md"

    runner = CliRunner()
    result = runner.invoke(auditml, ["report", "--results", str(results_dir), "--output", str(out)])

    assert result.exit_code == 0
    assert out.exists()
