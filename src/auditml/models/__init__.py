from .cnn import SimpleCNN
from .resnet import SmallResNet


def get_model(name: str, dataset: str):
    ds = dataset.lower()
    num_classes = 100 if ds == "cifar100" else 10
    input_channels = 1 if ds == "mnist" else 3
    input_size = 28 if ds == "mnist" else 32

    key = name.lower()
    if key in {"simple_cnn", "cnn"}:
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
    if key in {"resnet", "resnet18", "small_resnet"}:
        return SmallResNet(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
