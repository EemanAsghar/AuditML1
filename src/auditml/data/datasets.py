from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from .transforms import cifar_transform, mnist_transform


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    input_shape: tuple[int, int, int]
    num_train: int
    num_test: int


DATASET_META = {
    "mnist": DatasetInfo("mnist", 10, (1, 28, 28), 60_000, 10_000),
    "cifar10": DatasetInfo("cifar10", 10, (3, 32, 32), 50_000, 10_000),
    "cifar100": DatasetInfo("cifar100", 100, (3, 32, 32), 50_000, 10_000),
}


def get_dataset_info(name: str) -> DatasetInfo:
    key = name.lower()
    if key not in DATASET_META:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_META[key]


def get_dataset(name: str, train: bool = True, root: str = "data", augmentation: bool = False):
    n = name.lower()
    if n == "mnist":
        return MNIST(root=root, train=train, download=True, transform=mnist_transform())
    if n == "cifar10":
        return CIFAR10(root=root, train=train, download=True, transform=cifar_transform(train=train and augmentation))
    if n == "cifar100":
        return CIFAR100(root=root, train=train, download=True, transform=cifar_transform(train=train and augmentation))
    raise ValueError(f"Unsupported dataset: {name}")


def create_member_nonmember_split(dataset, member_ratio: float = 0.5, seed: int = 42):
    if not 0 < member_ratio < 1:
        raise ValueError("member_ratio must be in (0, 1)")
    n_member = int(len(dataset) * member_ratio)
    n_nonmember = len(dataset) - n_member
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_member, n_nonmember], generator=g)


def get_shadow_data_splits(dataset, n_shadows: int = 5, seed: int = 42):
    if n_shadows <= 0:
        raise ValueError("n_shadows must be > 0")
    return [create_member_nonmember_split(dataset, 0.5, seed + i) for i in range(n_shadows)]


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
    augmentation: bool = False,
    return_indices: bool = False,
):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")

    full = get_dataset(dataset_name, train=True, augmentation=augmentation)
    n_train = int(len(full) * train_ratio)
    n_val = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    test_set = get_dataset(dataset_name, train=False, augmentation=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_indices:
        train_indices = getattr(train_set, "indices", [])
        val_indices = getattr(val_set, "indices", [])
        return train_loader, val_loader, test_loader, train_indices, val_indices

    return train_loader, val_loader, test_loader
