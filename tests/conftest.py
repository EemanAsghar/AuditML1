import pytest

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    TensorDataset = None


@pytest.fixture
def sample_dataset():
    if torch is None:
        pytest.skip("torch not available")
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    return TensorDataset(x, y)


@pytest.fixture
def sample_loaders(sample_dataset):
    if torch is None:
        pytest.skip("torch not available")
    train_loader = DataLoader(sample_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(sample_dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def sample_model():
    if torch is None:
        pytest.skip("torch not available")
    from auditml.models.cnn import SimpleCNN

    return SimpleCNN(input_channels=1, num_classes=10, input_size=28)
