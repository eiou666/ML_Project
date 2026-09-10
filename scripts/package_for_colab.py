from __future__ import annotations

import zipfile
from pathlib import Path

IGNORE_DIRS = {
    ".venv",
    ".uv-cache",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "outputs",
    "tmp",
}
IGNORE_EXTS = {".pyc", ".pt"}


def main() -> None:
    root = Path.cwd()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ML_Project_Colab.zip"

    print(f"Creating clean Colab package: {archive_path}...")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if path.suffix in IGNORE_EXTS:
                continue
            if path.is_file():
                relative_path = Path("ML_Project") / path.relative_to(root)
                archive.write(path, relative_path)
                print(f"  Added: {relative_path}")

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully created {archive_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
