import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_rel_path: str):
    script = ROOT / script_rel_path
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc


def test_run_all_attacks_help_works_without_pythonpath():
    proc = _run_help("scripts/run_all_attacks.py")
    assert proc.returncode == 0, proc.stderr


def test_train_dp_models_help_works_without_pythonpath():
    proc = _run_help("scripts/train_dp_models.py")
    assert proc.returncode == 0, proc.stderr


def test_full_validation_help_works_without_pythonpath():
    proc = _run_help("scripts/full_validation.py")
    assert proc.returncode == 0, proc.stderr
