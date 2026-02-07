from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from auditml.attacks import get_attack
from auditml.config import load_config
from auditml.data import get_dataloaders
from auditml.models import get_model
from auditml.training import Trainer
from auditml.utils.device import get_device
from auditml.utils.reproducibility import set_seed


@click.group()
def auditml():
    """AuditML command line interface."""


@auditml.command()
def info():
    device = get_device()
    click.echo("AuditML version: 0.1.0")
    click.echo(f"Detected device: {device}")


@auditml.command()
@click.option("--dataset", default="mnist")
@click.option("--model", "model_name", default="simple_cnn")
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", default=64, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--seed", default=42, type=int)
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--output", required=True, type=click.Path())
@click.option("--save-members", is_flag=True, help="Save member indices next to model output")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def train(dataset, model_name, epochs, batch_size, lr, seed, train_ratio, output, save_members, config_path):
    set_seed(seed)
    device = get_device()

    if config_path:
        cfg = load_config(config_path, default_path="configs/default.yaml")
        dataset = cfg.data.dataset
        model_name = cfg.model.name
        epochs = cfg.train.epochs
        batch_size = cfg.train.batch_size
        lr = cfg.train.lr
        train_ratio = cfg.data.train_ratio

    train_loader, val_loader, _, train_indices, _ = get_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=seed,
        return_indices=True,
    )

    model = get_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), device)
    trainer.train(epochs=epochs)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    if save_members:
        np.save(out.with_suffix(".members.npy"), np.asarray(train_indices, dtype=np.int64))

    click.echo(f"Saved model to {out}")


@auditml.group()
def attack():
    """Run privacy attacks."""


@attack.command("mia-threshold")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", required=True, type=click.Path())
def mia_threshold(model_path, dataset, batch_size, output):
    device = get_device()
    train_loader, _, test_loader = get_dataloaders(dataset, batch_size=batch_size)
    model = get_model("simple_cnn", dataset).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    attack_instance = get_attack("mia-threshold", model, device=device)
    result = attack_instance.run(train_loader, test_loader)
    metrics = attack_instance.evaluate(result)
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "metrics.txt").write_text(str(metrics), encoding="utf-8")
    click.echo(metrics)


@attack.command("all")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--output", required=True, type=click.Path())
def attack_all(model_path, dataset, output):
    click.echo("Implemented: mia-threshold. Other attacks are scaffolded in src/auditml/attacks/.")


if __name__ == "__main__":
    auditml()
