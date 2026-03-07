"""Run all attacks for a trained model and store JSON outputs.

Usage:
  PYTHONPATH=src python scripts/run_all_attacks.py --model models/baseline_mnist.pt --dataset mnist --output-dir results/attacks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from auditml.attacks import get_attack
from auditml.data import get_dataloaders
from auditml.models import get_model
from auditml.utils.device import get_device


ATTACKS = ["mia-threshold", "mia-shadow", "inversion", "attribute"]


def run_attack(model_path: Path, dataset: str, attack_name: str, output_dir: Path, batch_size: int = 64):
    device = get_device()
    train_loader, _, test_loader = get_dataloaders(dataset, batch_size=batch_size)
    model = get_model("simple_cnn", dataset).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    attack_config = {"steps": 25, "max_classes": 5} if attack_name == "inversion" else {}
    attack = get_attack(attack_name, model, config=attack_config, device=device)
    result = attack.run(train_loader, test_loader)
    metrics = attack.evaluate(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{attack_name.replace('-', '_')}.json"
    out_file.write_text(json.dumps({"metrics": metrics, "metadata": result.metadata}, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--output-dir", default="results/attacks")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    summary = {}
    for attack_name in ATTACKS:
        print(f"[ATTACK] {attack_name}")
        summary[attack_name] = run_attack(model_path, args.dataset, attack_name, output_dir, args.batch_size)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
