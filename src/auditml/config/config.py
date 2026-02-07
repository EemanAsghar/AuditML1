from dataclasses import dataclass, asdict, field
import hashlib
import json
import yaml


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"


@dataclass
class ModelConfig:
    name: str = "simple_cnn"
    dataset: str = "mnist"


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
    dp: DPConfig = field(default_factory=DPConfig)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    c = Config()
    for section in ("train", "model", "dp"):
        if section in data:
            for k, v in data[section].items():
                setattr(getattr(c, section), k, v)
    validate_config(c)
    return c


def save_config(config: Config, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def validate_config(config: Config) -> None:
    if config.train.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if config.dp.enabled and config.dp.epsilon <= 0:
        raise ValueError("DP epsilon must be positive")
    if config.model.dataset not in {"mnist", "cifar10", "cifar100"}:
        raise ValueError("Invalid dataset")


def config_hash(config: Config) -> str:
    blob = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
