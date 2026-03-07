from pathlib import Path

from auditml.reporting.report import ReportGenerator


def test_report_generator_creates_markdown(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "sample.csv").write_text(
        "attack,dataset,accuracy\n"
        "mia-threshold,mnist,0.6\n"
        "mia-threshold,mnist,0.7\n",
        encoding="utf-8",
    )

    out = tmp_path / "report.md"
    generator = ReportGenerator()
    generator.generate_markdown_summary(str(results_dir), str(out))

    text = out.read_text(encoding="utf-8")
    assert "# AuditML Report" in text
    assert "mia-threshold" in text
