from pathlib import Path

import pytest

pytest.importorskip("yaml")

from auditml.config.config import Config, config_hash, load_config, save_config, validate_config


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.dataset == "mnist"


def test_invalid_learning_rate():
    cfg = Config()
    cfg.train.lr = -1
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_config_hash_deterministic():
    cfg = Config()
    assert config_hash(cfg) == config_hash(cfg)


def test_config_merge(tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text("model:\n  dataset: cifar10\ntrain:\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(override, default_path="configs/default.yaml")
    assert cfg.model.dataset == "cifar10"
    assert cfg.train.epochs == 5


def test_save_and_reload(tmp_path: Path):
    cfg = Config()
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    loaded = load_config(out)
    assert loaded.model.name == cfg.model.name
