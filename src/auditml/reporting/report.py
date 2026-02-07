from pathlib import Path
import pandas as pd


class ReportGenerator:
    def load_results(self, results_dir: str) -> pd.DataFrame:
        root = Path(results_dir)
        files = list(root.glob("**/*.csv"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    def generate_markdown_summary(self, results_dir: str, out_path: str):
        df = self.load_results(results_dir)
        lines = ["# AuditML Report", ""]
        if df.empty:
            lines.append("No CSV results found.")
        else:
            lines.append(df.groupby(["attack", "dataset"]).mean(numeric_only=True).to_markdown())
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
