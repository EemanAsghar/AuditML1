from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from .transforms import mnist_transform, cifar_transform


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]
    num_train: int
    num_test: int


def get_dataset(name: str, train: bool = True, root: str = "data"):
    n = name.lower()
    if n == "mnist":
        return MNIST(root=root, train=train, download=True, transform=mnist_transform())
    if n == "cifar10":
        return CIFAR10(root=root, train=train, download=True, transform=cifar_transform(train=train))
    if n == "cifar100":
        return CIFAR100(root=root, train=train, download=True, transform=cifar_transform(train=train))
    raise ValueError(f"Unsupported dataset: {name}")


def create_member_nonmember_split(dataset, member_ratio: float = 0.5, seed: int = 42):
    n_member = int(len(dataset) * member_ratio)
    n_nonmember = len(dataset) - n_member
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_member, n_nonmember], generator=g)


def get_shadow_data_splits(dataset, n_shadows: int = 5, seed: int = 42):
    splits = []
    for i in range(n_shadows):
        splits.append(create_member_nonmember_split(dataset, 0.5, seed + i))
    return splits


def get_dataloaders(dataset_name: str, batch_size: int = 64, train_ratio: float = 0.8, seed: int = 42):
    full = get_dataset(dataset_name, train=True)
    n_train = int(len(full) * train_ratio)
    n_val = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    test_set = get_dataset(dataset_name, train=False)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
