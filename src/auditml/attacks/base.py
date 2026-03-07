from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


@dataclass
class AttackResult:
    predictions: np.ndarray
    ground_truth: np.ndarray
    confidence_scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    def __init__(self, target_model, config=None, device=None):
        self.target_model = target_model
        self.config = config or {}
        self.device = device or torch.device("cpu")

    @abstractmethod
    def run(self, member_data, nonmember_data):
        raise NotImplementedError

    def get_model_outputs(self, loader):
        self.target_model.eval()
        logits_all, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                logits = self.target_model(x)
                probs = F.softmax(logits, dim=1)
                logits_all.append(logits.detach().cpu())
                probs_all.append(probs.detach().cpu())
                labels_all.append(y.detach().cpu())
        return torch.cat(logits_all), torch.cat(probs_all), torch.cat(labels_all)

    def get_loss_values(self, loader):
        self.target_model.eval()
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.target_model(x)
                batch_loss = F.cross_entropy(logits, y, reduction="none")
                losses.extend(batch_loss.detach().cpu().numpy())
        return np.asarray(losses)

    @staticmethod
    def _compute_metrics(y_true, y_pred, scores):
        unique = np.unique(y_true)
        average = "binary" if len(unique) <= 2 else "macro"
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        if len(unique) < 2:
            out["auc_roc"] = 0.5
        else:
            try:
                out["auc_roc"] = float(roc_auc_score(y_true, scores))
            except ValueError:
                out["auc_roc"] = 0.5
        return out

    def evaluate(self, result: AttackResult):
        return self._compute_metrics(result.ground_truth, result.predictions, result.confidence_scores)
