from __future__ import annotations

import importlib
import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


REQUIRED = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "yaml",
    "pytest",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        raise RuntimeError(f"Python 3.11-3.13 is required, found {platform.python_version()}")
    modules = {name: importlib.import_module(name) for name in REQUIRED}
    torch = modules["torch"]
    torchvision = modules["torchvision"]
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local formal experiments")

    model = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1)
    sample = torch.randn(2, 1, 28, 28)
    output = model(sample)
    if output.shape != (2, 4, 28, 28):
        raise RuntimeError("PyTorch tensor smoke test failed")
    if torch.cuda.is_available():
        gpu_model = model.to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            gpu_loss = gpu_model(sample.to("cuda")).square().mean()
        gpu_loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(gpu_loss):
            raise RuntimeError("CUDA AMP forward/backward test failed")

    from sklearn.cluster import kmeans_plusplus

    _, selected = kmeans_plusplus(modules["numpy"].eye(6), n_clusters=2, random_state=0)
    if len(selected) != 2:
        raise RuntimeError("scikit-learn k-means++ smoke test failed")

    output_dir = Path("outputs") / "environment-check"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "write-test.txt"
    marker.write_text("ok\n", encoding="utf-8")
    if marker.read_text(encoding="utf-8").strip() != "ok":
        raise RuntimeError("Workspace write test failed")

    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": modules["numpy"].__version__,
        "pandas": modules["pandas"].__version__,
        "scipy": modules["scipy"].__version__,
        "scikit_learn": modules["sklearn"].__version__,
        "pytest": modules["pytest"].__version__,
        "workspace_write": True,
    }
    (output_dir / "environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
