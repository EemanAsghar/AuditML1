import torch.nn as nn


def get_loss(name: str = "cross_entropy"):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")
