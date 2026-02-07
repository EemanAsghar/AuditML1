from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_features(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
