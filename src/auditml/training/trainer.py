from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        grad_clip_norm: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    @classmethod
    def from_config(cls, model, train_loader, val_loader, config, device):
        lr = config.train.lr
        optimizer_name = config.train.optimizer.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {config.train.optimizer}")
        criterion = torch.nn.CrossEntropyLoss()
        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=config.train.grad_clip_norm,
        )

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(self.train_loader, desc="train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}

    def train(self, epochs: int, patience: int = 10, checkpoint_dir: str | None = None):
        best_loss, wait = float("inf"), 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2)
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch()
            val = self.evaluate(self.val_loader)
            scheduler.step(val["loss"])
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val["loss"])
            self.history["val_acc"].append(val["acc"])
            if val["loss"] < best_loss:
                best_loss, wait = val["loss"], 0
                if ckpt_dir:
                    self.save_checkpoint(ckpt_dir / "best.pt", epoch, val)
            else:
                wait += 1
            if wait >= patience:
                break
        return self.history

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, Any]):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        return ckpt
