"""Run a compact validation matrix across datasets/privacy levels/attacks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from run_all_attacks import ATTACKS, run_attack
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_all_attacks import ATTACKS, run_attack


def resolve_model_path(dataset: str, kind: str, model_pattern: str | None = None) -> Path | None:
    candidates: list[Path] = []

    if model_pattern:
        candidates.append(Path(model_pattern.format(kind=kind, dataset=dataset)))

    if kind == "baseline":
        candidates.extend(
            [
                Path(f"models/baselines/{dataset}/model.pt"),
                Path(f"models/baseline_{dataset}.pt"),
            ]
        )
    elif kind.startswith("dp_eps"):
        eps = kind.replace("dp_eps", "")
        candidates.extend(
            [
                Path(f"models/dp/{dataset}_eps{eps}.pt"),
                Path(f"models/{kind}_{dataset}.pt"),
            ]
        )
    else:
        candidates.append(Path(f"models/{kind}_{dataset}.pt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    parser.add_argument(
        "--model-pattern",
        default=None,
        help="optional explicit template supporting {kind} and {dataset}",
    )
    parser.add_argument("--kinds", nargs="+", default=["baseline", "dp_eps10", "dp_eps1", "dp_eps0.1"])
    parser.add_argument("--output-dir", default="results/validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for dataset in args.datasets:
        for kind in args.kinds:
            model_path = resolve_model_path(dataset, kind, args.model_pattern)
            if model_path is None:
                print(f"[SKIP] missing model for dataset={dataset} kind={kind}")
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
