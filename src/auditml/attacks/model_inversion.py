import torch
from .base import BaseAttack


class ModelInversion(BaseAttack):
    def invert_class(self, target_class: int, shape=(1, 1, 28, 28), steps: int = 500, lr: float = 0.1):
        x = torch.rand(shape, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = self.target_model(x)
            loss = -torch.nn.functional.log_softmax(logits, dim=1)[0, target_class]
            loss.backward()
            optimizer.step()
            x.data.clamp_(0, 1)
        return x.detach().cpu()

    def run(self, member_data, nonmember_data):
        return {"message": "Use invert_class for per-class reconstruction."}
