"""Train differential privacy models for multiple datasets and epsilon levels.

Usage:
  python scripts/train_dp_models.py --epochs 1 --output-dir models/dp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from auditml.data import get_dataloaders
from auditml.models import get_model
from auditml.utils.device import get_device
from auditml.utils.reproducibility import set_seed
from auditml.defenses.dp_training import DPTrainer


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_dp_model(dataset: str, epsilon: float, epochs: int, output_dir: Path, seed: int = 42):
    noise_map = {10.0: 0.5, 1.0: 1.5, 0.1: 5.0}
    noise = noise_map.get(float(epsilon), 1.5)

    set_seed(seed)
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(dataset, batch_size=64, seed=seed)
    model = get_model("simple_cnn", dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dp = DPTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        noise_multiplier=noise,
        max_grad_norm=1.0,
    )

    criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(dp.model, dp.train_loader, dp.optimizer, criterion, device)
        val_loss, val_acc = _evaluate(dp.model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epsilon_spent": float(dp.epsilon(delta=1e-5)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{dataset}_eps{epsilon}.pt"
    torch.save(dp.model.state_dict(), model_path)

    metrics_path = output_dir / f"{dataset}_eps{epsilon}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "target_epsilon": epsilon,
                "noise_multiplier": noise,
                "final": history[-1],
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[10.0, 1.0, 0.1])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-dir", default="models/dp")
    args = parser.parse_args()

    out = Path(args.output_dir)
    for dataset in args.datasets:
        for eps in args.epsilons:
            print(f"[DP] training dataset={dataset}, epsilon={eps}")
            train_dp_model(dataset, eps, args.epochs, out)


if __name__ == "__main__":
    main()
