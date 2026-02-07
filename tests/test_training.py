from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from auditml.config.config import Config
from auditml.training import Trainer


def test_train_one_epoch(sample_model, sample_loaders):
    device = torch.device("cpu")
    train_loader, val_loader = sample_loaders
    model = sample_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), device)
    history = trainer.train(epochs=1, patience=1)
    assert len(history["train_loss"]) == 1


def test_checkpoint_save_load(sample_model, sample_loaders, tmp_path: Path):
    device = torch.device("cpu")
    train_loader, val_loader = sample_loaders
    model = sample_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), device)
    trainer.train(epochs=1, patience=1)

    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt, 1, {"loss": 1.0, "acc": 0.5})
    loaded = trainer.load_checkpoint(ckpt)
    assert loaded["epoch"] == 1


def test_from_config(sample_model, sample_loaders):
    device = torch.device("cpu")
    cfg = Config()
    cfg.train.optimizer = "adam"
    train_loader, val_loader = sample_loaders
    trainer = Trainer.from_config(sample_model.to(device), train_loader, val_loader, cfg, device)
    assert isinstance(trainer.optimizer, torch.optim.Adam)
