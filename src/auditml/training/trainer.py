from pathlib import Path
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(self.train_loader, desc="train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

    def save_checkpoint(self, path, epoch, metrics):
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "metrics": metrics}, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        return ckpt
