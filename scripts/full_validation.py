"""Run a compact full validation matrix across datasets/privacy levels/attacks.

This is an MVP orchestration script for repeatable experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_all_attacks import run_attack, ATTACKS


def _read_metric_file(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--model-pattern", default="models/{kind}_{dataset}.pt", help="supports {kind} and {dataset}")
    parser.add_argument("--kinds", nargs="+", default=["baseline", "dp_eps10", "dp_eps1", "dp_eps0.1"])
    parser.add_argument("--output-dir", default="results/validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        for kind in args.kinds:
            model_path = Path(args.model_pattern.format(kind=kind, dataset=dataset))
            if not model_path.exists():
                print(f"[SKIP] missing model: {model_path}")
                continue

            exp_dir = output_dir / dataset / kind
            exp_dir.mkdir(parents=True, exist_ok=True)

            for attack_name in ATTACKS:
                print(f"[RUN] dataset={dataset} kind={kind} attack={attack_name}")
                metrics = run_attack(model_path, dataset, attack_name, exp_dir / attack_name)
                row = {"dataset": dataset, "model_kind": kind, "attack": attack_name}
                row.update({k: float(v) for k, v in metrics.items()})
                rows.append(row)

    csv_path = output_dir / "validation_summary.csv"
    if rows:
        fields = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows -> {csv_path}")
    else:
        print("No rows generated; provide existing model files first.")


if __name__ == "__main__":
    main()
