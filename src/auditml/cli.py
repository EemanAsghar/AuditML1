from pathlib import Path
import json

import click
import torch

from auditml.attacks import get_attack
from auditml.data import get_dataloaders
from auditml.models import get_model
from auditml.training import Trainer, get_loss
from auditml.utils.device import get_device
from auditml.utils.reproducibility import set_seed
from auditml.reporting.report import ReportGenerator


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
@click.option("--output", required=True, type=click.Path())
def train(dataset, model_name, epochs, batch_size, lr, seed, output):
    set_seed(seed)
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(dataset, batch_size=batch_size, seed=seed)
    model = get_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, val_loader, optimizer, get_loss(), device)
    trainer.train(epochs=epochs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    click.echo(f"Saved model to {out}")


def _load_target_model(model_path: str, dataset: str, device):
    model = get_model("simple_cnn", dataset).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def _run_single_attack(attack_name: str, model_path: str, dataset: str, batch_size: int, output: str):
    device = get_device()
    train_loader, _, test_loader = get_dataloaders(dataset, batch_size=batch_size)
    model = _load_target_model(model_path, dataset, device)
    config = {"steps": 20, "max_classes": 5} if attack_name == "inversion" else {}
    attack = get_attack(attack_name, model, config=config, device=device)
    result = attack.run(train_loader, test_loader)
    metrics = attack.evaluate(result)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "metadata": result.metadata}
    (out_dir / f"{attack_name.replace('-', '_')}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


@auditml.group()
def attack():
    """Run privacy attacks."""


@attack.command("mia-threshold")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", required=True, type=click.Path())
def mia_threshold(model_path, dataset, batch_size, output):
    metrics = _run_single_attack("mia-threshold", model_path, dataset, batch_size, output)
    click.echo(metrics)


@attack.command("mia-shadow")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", required=True, type=click.Path())
def mia_shadow(model_path, dataset, batch_size, output):
    metrics = _run_single_attack("mia-shadow", model_path, dataset, batch_size, output)
    click.echo(metrics)


@attack.command("inversion")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=32, type=int)
@click.option("--output", required=True, type=click.Path())
def inversion(model_path, dataset, batch_size, output):
    metrics = _run_single_attack("inversion", model_path, dataset, batch_size, output)
    click.echo(metrics)


@attack.command("attribute")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", required=True, type=click.Path())
def attribute(model_path, dataset, batch_size, output):
    metrics = _run_single_attack("attribute", model_path, dataset, batch_size, output)
    click.echo(metrics)


@attack.command("all")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--data", "dataset", default="mnist")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", required=True, type=click.Path())
def attack_all(model_path, dataset, batch_size, output):
    summary = {}
    for name in ["mia-threshold", "mia-shadow", "inversion", "attribute"]:
        summary[name] = _run_single_attack(name, model_path, dataset, batch_size, output)
    click.echo(json.dumps(summary, indent=2))


@auditml.command("report")
@click.option("--results", "results_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
def report(results_dir, output_path):
    """Generate a markdown summary report from CSV results."""
    generator = ReportGenerator()
    generator.generate_markdown_summary(results_dir, output_path)
    click.echo(f"Saved report to {output_path}")


if __name__ == "__main__":
    auditml()
