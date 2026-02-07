from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def setup_logging(log_dir: str, experiment_name: str = "run") -> logging.Logger:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(path / "training.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class ExperimentLogger:
    def __init__(self, root_dir: str, experiment_name: str):
        self.base = Path(root_dir) / experiment_name
        self.logs_dir = self.base / "logs"
        self.metrics_dir = self.base / "metrics"
        self.checkpoints_dir = self.base / "checkpoints"
        for d in (self.logs_dir, self.metrics_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(str(self.logs_dir), experiment_name)
        self.metrics_rows: list[dict[str, Any]] = []

    def log_metrics(self, metrics_dict: dict[str, Any], step: int) -> None:
        row = {"step": step, **metrics_dict}
        self.metrics_rows.append(row)
        self.logger.info("metrics step=%s %s", step, metrics_dict)
        pd.DataFrame(self.metrics_rows).to_csv(self.metrics_dir / "metrics.csv", index=False)

    def log_config(self, config_dict: dict[str, Any]) -> None:
        with open(self.base / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    def log_system_info(self) -> None:
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        with open(self.base / "system_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def log_model_summary(self, model) -> None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info("model params total=%s trainable=%s", total, trainable)
