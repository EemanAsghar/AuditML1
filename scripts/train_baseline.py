from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from auditml.config import load_config, save_config
from auditml.data import get_dataloaders
from auditml.models import get_model
from auditml.training import Trainer
from auditml.utils.device import get_device
from auditml.utils.reproducibility import set_seed


def _plot_history(history: dict, output_file: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def train_one_dataset(config_path: str, default_path: str, output_dir: str, seed: int) -> tuple[str, float]:
    set_seed(seed)
    cfg = load_config(config_path, default_path=default_path)
    device = get_device()

    train_loader, val_loader, test_loader, train_indices, _ = get_dataloaders(
        cfg.data.dataset,
        batch_size=cfg.train.batch_size,
        train_ratio=cfg.data.train_ratio,
        seed=seed,
        num_workers=cfg.data.num_workers,
        augmentation=cfg.data.augmentation,
        return_indices=True,
    )

    model = get_model(cfg.model.name, cfg.model.dataset).to(device)
    trainer = Trainer.from_config(model, train_loader, val_loader, cfg, device)
    history = trainer.train(
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        checkpoint_dir=cfg.train.checkpoint_dir,
    )
    test_metrics = trainer.evaluate(test_loader)

    out = Path(output_dir) / cfg.data.dataset
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    save_config(cfg, out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps({"history": history, "test": test_metrics}, indent=2), encoding="utf-8")
    np.save(out / "members.npy", np.asarray(train_indices, dtype=np.int64))
    _plot_history(history, out / "training_curves.png")

    print(f"Saved baseline artifacts in {out}")
    print(f"Test metrics: {test_metrics}")
    return cfg.data.dataset, float(test_metrics["acc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model(s) and save artifacts")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="models/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-datasets", action="store_true", help="Train mnist, cifar10, cifar100 sequentially")
    args = parser.parse_args()

    results: list[tuple[str, float]] = []
    if args.all_datasets:
        experiment_map = {
            "mnist": "configs/experiments/mnist_baseline.yaml",
            "cifar10": "configs/experiments/cifar10_baseline.yaml",
            "cifar100": "configs/experiments/cifar100_baseline.yaml",
        }
        for ds, cfg in experiment_map.items():
            try:
                results.append(train_one_dataset(cfg, args.default_config, args.output_dir, args.seed))
            except Exception as exc:  # pragma: no cover - best effort batch mode
                print(f"Failed dataset {ds}: {exc}")
    else:
        results.append(train_one_dataset(args.config, args.default_config, args.output_dir, args.seed))

    summary_lines = ["# Baseline Model Accuracies", ""]
    for dataset, acc in results:
        summary_lines.append(f"- {dataset}: {acc:.4f}")
    summary_path = Path(args.output_dir) / "README.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
