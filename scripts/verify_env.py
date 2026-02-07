import importlib
import platform


def _optional_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _check_imports() -> dict[str, bool]:
    modules = ["torch", "torchvision", "numpy", "sklearn", "opacus", "yaml", "click", "tqdm"]
    return {m: _optional_import(m) is not None for m in modules}


def main() -> None:
    torch = _optional_import("torch")
    numpy = _optional_import("numpy")

    print("Python:", platform.python_version())
    if torch is None:
        print("PyTorch: MISSING")
    else:
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA version:", torch.version.cuda)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print("Tensor check:", y.tolist())

    if numpy is not None:
        print("NumPy:", numpy.__version__)

    print("Import checks:")
    for mod, ok in _check_imports().items():
        print(f"  - {mod}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()
