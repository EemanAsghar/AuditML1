import pytest

torch = pytest.importorskip("torch")

from auditml.models import get_model


def test_simple_cnn_output_shape(sample_model):
    x = torch.randn(8, 1, 28, 28)
    out = sample_model(x)
    assert out.shape == (8, 10)


def test_simple_cnn_features_shape(sample_model):
    x = torch.randn(8, 1, 28, 28)
    features = sample_model.get_features(x)
    assert features.shape == (8, 128)


def test_model_factory_mnist():
    model = get_model("simple_cnn", "mnist")
    x = torch.randn(4, 1, 28, 28)
    assert model(x).shape == (4, 10)
