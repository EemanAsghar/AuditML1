import torch
import torch.nn as nn
from .base import BaseModel


class SimpleCNN(BaseModel):
    def __init__(self, input_channels: int = 1, num_classes: int = 10, input_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        spatial = input_size // 4
        self.fc1 = nn.Linear(64 * spatial * spatial, 128)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.get_features(x))
