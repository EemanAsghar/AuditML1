from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "plateau"
    patience: int = 10
    checkpoint_dir: str = "models/checkpoints"
    grad_clip_norm: float = 1.0


@dataclass
class ModelConfig:
    name: str = "simple_cnn"
    dataset: str = "mnist"


@dataclass
class DataConfig:
    dataset: str = "mnist"
    train_ratio: float = 0.8
    augmentation: bool = False
    num_workers: int = 0


@dataclass
class AttackConfig:
    attack_type: str = "mia_threshold"


@dataclass
class DPConfig:
    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    dp: DPConfig = field(default_factory=DPConfig)


def _apply_section(section_obj: Any, section_data: dict[str, Any]) -> None:
    for key, value in section_data.items():
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)


def load_config(path: str | Path, default_path: str | Path | None = None) -> Config:
    config = Config()
    if default_path:
        with open(default_path, "r", encoding="utf-8") as f:
            default_data = yaml.safe_load(f) or {}
        _merge_into_config(config, default_data)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _merge_into_config(config, data)

    validate_config(config)
    return config


def _merge_into_config(config: Config, data: dict[str, Any]) -> None:
    for section_name in ("train", "model", "data", "attack", "dp"):
        if section_name in data:
            _apply_section(getattr(config, section_name), data[section_name])


def save_config(config: Config, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def validate_config(config: Config) -> None:
    if config.train.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if config.train.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if config.train.epochs <= 0:
        raise ValueError("Epochs must be positive")
    if not 0 < config.data.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if config.dp.enabled and config.dp.epsilon <= 0:
        raise ValueError("DP epsilon must be positive")

    valid_datasets = {"mnist", "cifar10", "cifar100"}
    if config.model.dataset not in valid_datasets:
        raise ValueError("Invalid model dataset")
    if config.data.dataset not in valid_datasets:
        raise ValueError("Invalid data dataset")


def config_hash(config: Config) -> str:
    blob = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
