from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


@dataclass
class AttackResult:
    predictions: np.ndarray
    ground_truth: np.ndarray
    confidence_scores: np.ndarray
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):
    def __init__(self, target_model, config=None, device=None):
        self.target_model = target_model
        self.config = config or {}
        self.device = device or torch.device("cpu")

    @abstractmethod
    def run(self, member_data, nonmember_data):
        raise NotImplementedError

    def evaluate(self, result: AttackResult):
        y_true = result.ground_truth
        y_pred = result.predictions
        scores = result.confidence_scores
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        out = {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1}
        try:
            out["auc_roc"] = roc_auc_score(y_true, scores)
        except ValueError:
            out["auc_roc"] = 0.5
        return out
