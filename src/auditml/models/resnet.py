import torch.nn as nn
from torchvision.models import resnet18
from .base import BaseModel


class SmallResNet(BaseModel):
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet18(num_classes=num_classes)
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def get_features(self, x):
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.backbone(x)
