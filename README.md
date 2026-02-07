# AuditML

Comprehensive Privacy Auditing Toolkit for PyTorch Models.

AuditML helps teams evaluate privacy leakage risks in machine learning models using four attack families:

- Membership Inference (Threshold)
- Membership Inference (Shadow Models)
- Model Inversion
- Attribute Inference

## Installation

```bash
pip install -e .
```

## Quickstart (Phase 1)

```bash
python scripts/verify_env.py
python scripts/train_baseline.py --config configs/experiments/mnist_baseline.yaml
```

To run all baseline datasets (MNIST/CIFAR-10/CIFAR-100):

```bash
python scripts/train_baseline.py --all-datasets
```

## Project Structure

- `src/auditml/`: main package
- `configs/`: YAML experiment configs
- `scripts/`: helper scripts
- `tests/`: unit tests
- `docs/`: documentation
- `models/baselines/`: baseline artifacts

## Phase 1 Completion Checklist

- [x] Project setup and package structure
- [x] Device + reproducibility utilities
- [x] Base model architectures and factory
- [x] Dataset loading, preprocessing, and split helpers
- [x] Reusable training loop with checkpointing
- [x] Config management with validation + hashing
- [x] Logging + metrics utility
- [x] Unit test framework and core tests
- [x] Baseline training script + artifact persistence (model/config/metrics/members/curves)

## Note

Phase 2+ attack pipelines are still in progress.
