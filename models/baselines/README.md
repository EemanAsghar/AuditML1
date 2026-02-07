# Baseline Models

This directory stores trained non-private baseline model artifacts produced by:

```bash
python scripts/train_baseline.py --all-datasets
```

Expected artifacts per dataset subdirectory:
- `model.pt`
- `config.yaml`
- `metrics.json`
- `members.npy`
- `training_curves.png`

`README.md` in this folder is automatically refreshed by the training script with observed test accuracies.
