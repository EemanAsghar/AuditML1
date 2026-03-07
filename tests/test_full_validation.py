from pathlib import Path

from scripts.full_validation import resolve_model_path


def test_resolve_model_path_prefers_existing_baseline(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models" / "baselines" / "mnist" / "model.pt"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")

    resolved = resolve_model_path("mnist", "baseline")
    assert resolved == Path("models/baselines/mnist/model.pt")


def test_resolve_model_path_for_dp_kind(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models" / "dp" / "mnist_eps1.pt"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")

    resolved = resolve_model_path("mnist", "dp_eps1")
    assert resolved == Path("models/dp/mnist_eps1.pt")
