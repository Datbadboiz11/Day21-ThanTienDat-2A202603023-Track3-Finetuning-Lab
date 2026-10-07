#!/usr/bin/env python3
"""Build a portable input ZIP or a snapshot of completed Colab artifacts.

No model weights, optimizer checkpoints, caches, .git or secrets are included.
Progress archives contain all completed adapters and can be uploaded to resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "lab21_2A202603023"
SOURCE_DIRS = ("src", "scripts", "tests", "notebooks", "colab", "submission", "docs")
ROOT_FILES = ("README.md", "rubric.md", "HARDWARE-GUIDE.md", "COLAB_GUIDE.md",
              "requirements.txt", "requirements-cpu.txt", "pyproject.toml", ".env.example",
              "LICENSE", "Makefile", "SIMULATION-FINDINGS.md")
DATA_FILES = ("train_seed.jsonl", "eval_target.jsonl", "eval_regression.jsonl", "checksums.json")
SUFFIXES = {".py", ".ipynb", ".md", ".tex", ".json", ".txt", ".png"}


def project_files(root: pathlib.Path, mode: str) -> list[pathlib.Path]:
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in SOURCE_DIRS:
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in SUFFIXES
                     and "__pycache__" not in p.parts and not p.is_symlink())
    for name in DATA_FILES:
        p = root / "data" / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing required data: {p}")
        files.append(p)
    custom = root / "data" / "CUSTOM_DATASET.md"
    if custom.is_file():
        files.append(custom)
    if mode == "progress":
        files.extend(p for p in (root / "results").glob("*")
                     if p.is_file() and p.suffix in {".json", ".csv", ".txt"})
        files.extend((root / "data" / "split").glob("*.jsonl"))
        for run in ("correct", "attn_only", "wrong_lr", "qlora"):
            # Only the saved adapter/tokenizer, never checkpoint-* optimizer state.
            files.extend(p for p in (root / "adapters" / run).glob("*")
                         if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model"})
    return sorted(set(files))


def build_archive(root: pathlib.Path, output: pathlib.Path, mode: str) -> pathlib.Path:
    files = project_files(root, mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    manifest = {"mode": mode, "student_id": "2A202603023", "files": {}}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"File is outside project: {path}")
            relative = path.relative_to(root).as_posix()
            archive.write(path, f"{ARCHIVE_ROOT}/{relative}")
            manifest["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive.writestr(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(output)
    return output


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("input", "progress"), default="input")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    name = "COLAB_INPUT" if args.mode == "input" else "LAB21_RESULTS"
    output = args.output or ROOT / "dist" / f"{name}_2A202603023.zip"
    path = build_archive(ROOT, output, args.mode)
    with zipfile.ZipFile(path) as archive:
        print(f"{path} — {len(archive.namelist())} files, {path.stat().st_size / 1024**2:.2f} MB")
    if args.mode == "progress":
        print("Snapshot for backup/report writing; submission readiness is checked by scripts/verify.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
