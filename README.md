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

## Quickstart

```bash
auditml info
auditml train --dataset mnist --epochs 1 --output models/baseline_mnist.pt
auditml attack mia-threshold --model models/baseline_mnist.pt --data mnist --output results/mia_threshold
```

## Project Structure

- `src/auditml/`: main package
- `configs/`: YAML experiment configs
- `scripts/`: helper scripts
- `tests/`: test suite skeleton
- `docs/`: documentation

## Status

Initial end-to-end scaffold is included with CLI, training, attacks, and report module stubs to support incremental development.
