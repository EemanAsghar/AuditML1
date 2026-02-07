import numpy as np
import torch
import torch.nn.functional as F
from .base import BaseAttack, AttackResult


class ThresholdMIA(BaseAttack):
    def _collect_losses(self, loader):
        self.target_model.eval()
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.target_model(x)
                batch = F.cross_entropy(logits, y, reduction="none")
                losses.extend(batch.detach().cpu().numpy())
        return np.asarray(losses)

    def run(self, member_data, nonmember_data):
        m = self._collect_losses(member_data)
        n = self._collect_losses(nonmember_data)
        scores = np.concatenate([-m, -n])
        y_true = np.concatenate([np.ones(len(m)), np.zeros(len(n))]).astype(int)
        threshold = np.median(np.concatenate([m, n]))
        y_pred = (np.concatenate([m, n]) < threshold).astype(int)
        return AttackResult(y_pred, y_true, scores, metadata={"threshold": float(threshold)})
