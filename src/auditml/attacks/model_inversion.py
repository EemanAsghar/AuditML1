import numpy as np
import torch

from .base import AttackResult, BaseAttack


class ModelInversion(BaseAttack):
    def invert_class(self, target_class: int, shape=(1, 1, 28, 28), steps: int = 150, lr: float = 0.1):
        x = torch.rand(shape, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)
        last_prob = 0.0
        for _ in range(steps):
            optimizer.zero_grad()
            logits = self.target_model(x)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -log_probs[0, target_class]
            loss.backward()
            optimizer.step()
            x.data.clamp_(0, 1)
            last_prob = float(log_probs.exp()[0, target_class].detach().cpu())
        return x.detach().cpu(), last_prob

    def run(self, member_data, nonmember_data):
        _, probs, _ = self.get_model_outputs(member_data)
        n_classes = probs.shape[1]
        input_shape = tuple(member_data.dataset[0][0].shape)

        recon_scores = []
        recon_samples = {}
        for class_id in range(min(n_classes, self.config.get("max_classes", n_classes))):
            recon, score = self.invert_class(
                class_id,
                shape=(1, *input_shape),
                steps=int(self.config.get("steps", 100)),
                lr=float(self.config.get("lr", 0.1)),
            )
            recon_scores.append(score)
            recon_samples[class_id] = recon.squeeze(0).numpy().tolist()

        y_true = np.ones(len(recon_scores), dtype=int)
        y_pred = (np.array(recon_scores) > 0.5).astype(int)
        return AttackResult(
            predictions=y_pred,
            ground_truth=y_true,
            confidence_scores=np.asarray(recon_scores),
            metadata={"reconstructions": recon_samples},
        )
