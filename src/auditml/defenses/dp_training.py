from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class DPTrainer:
    def __init__(self, model, optimizer, train_loader, noise_multiplier: float, max_grad_norm: float):
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    def epsilon(self, delta: float) -> float:
        return self.privacy_engine.get_epsilon(delta)
