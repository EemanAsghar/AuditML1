# AuditML Setup

## 1) Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2) Install package in editable mode

```bash
pip install -U pip
pip install -e .
```

## 3) Verify environment and imports

```bash
python scripts/verify_env.py
```

The script reports Python/PyTorch/CUDA details and checks imports for:
`torch`, `torchvision`, `numpy`, `sklearn`, `opacus`, `yaml`, `click`, `tqdm`.

## 4) Run tests

```bash
PYTHONPATH=src pytest
```

## 5) Train baseline model

```bash
python scripts/train_baseline.py --config configs/experiments/mnist_baseline.yaml
```
