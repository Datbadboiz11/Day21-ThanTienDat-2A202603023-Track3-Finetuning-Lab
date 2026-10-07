#!/usr/bin/env python3
"""Verify and package Option A (with main adapter) and Option C (without weights).

Original measurements and all four local adapters are preserved. Both deliverables
include report, all results, source, tests and the three public datasets to reproduce
checks. Option C uses exact recorded core dependency versions in requirements.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFIX = "lab21_2A202603023"


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(option: str) -> list[pathlib.Path]:
    from export_colab import project_files
    files = project_files(ROOT, "input")
    files.extend(path for path in (ROOT / "results").glob("*")
                 if path.is_file() and path.suffix in {".json", ".csv", ".txt"})
    files.extend((ROOT / "data" / "split").glob("*.jsonl"))
    files.extend((ROOT / "submission" / "figures").glob("*.png"))
    files.append(ROOT / "requirements-reproduce.txt")
    if option == "A":
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = ROOT / "adapters" / "correct" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    return sorted(set(files))


def write_archive(option: str, files: list[pathlib.Path]) -> pathlib.Path:
    destination = ROOT / "dist" / f"LAB21_SUBMISSION_2A202603023_OPTION_{option}.zip"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    manifest = {"student": "Thân Tiến Đạt", "student_id": "2A202603023",
                "format": option, "experiment_verdict": "FAILED",
                "limitations": ["Only one target ticket loses to baseline (b).",
                                "Additional observed losses are in the regression group.",
                                "Optional bonus experiments were not performed."], "files": {}}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"Unsafe file: {path}")
            name = path.relative_to(ROOT).as_posix()
            if option == "C" and name == "requirements.txt":
                content = (ROOT / "requirements-reproduce.txt").read_bytes()
                archive.writestr(f"{PREFIX}/{name}", content)
                manifest["files"][name] = hashlib.sha256(content).hexdigest()
            else:
                archive.write(path, f"{PREFIX}/{name}")
                manifest["files"][name] = sha(path)
        archive.writestr(f"{PREFIX}/SUBMISSION_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"Archive CRC failed: {bad}")
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256()
            with archive.open(f"{PREFIX}/{name}") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Archive checksum failed: {name}")
    temporary.replace(destination)
    print(f"Option {option}: {destination.name}, {destination.stat().st_size / 1024**2:.2f} MiB")
    return destination


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=("A", "C", "both"), default="both")
    args = parser.parse_args()
    temp_root = ROOT / ".pytest_cache"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission-check-", dir=temp_root) as directory:
        env = {**os.environ, "TEMP": directory, "TMP": directory,
               "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.run([sys.executable, "-X", "utf8", "scripts/verify.py"],
                                 cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    print(process.stdout)
    if process.returncode:
        print(process.stderr, file=sys.stderr)
        print("Submission not packaged: fix verify failures first.", file=sys.stderr)
        return process.returncode
    (ROOT / "submission/VERIFICATION.txt").write_text(
        "Command: python scripts/verify.py\n\n" + process.stdout
        + "\nThis checks artifacts and consistency, not a guarantee of full rubric marks.\n",
        encoding="utf-8")
    for option in ("A", "C") if args.option == "both" else (args.option,):
        write_archive(option, package_files(option))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
