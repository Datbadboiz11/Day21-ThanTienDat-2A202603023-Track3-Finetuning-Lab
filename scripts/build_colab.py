#!/usr/bin/env python3
"""Generate colab/*.ipynb from notebooks/*.py (jupytext py:percent).

The .py files are the source of truth. The Colab notebooks are generated, plus a
bootstrap cell that clones the repo and installs deps — Colab starts with no repo.

Usage: python scripts/build_colab.py    (standard library only)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks"
OUT = ROOT / "colab"

BOOTSTRAP = '''# @title 1. Setup — upload gói ZIP + cài dependency
import os, pathlib, subprocess, sys, zipfile

SOURCE = "upload"  # @param ["upload", "github"]
SAVE_TO_DRIVE = True  # @param {type:"boolean"}
REPO = "https://github.com/Datbadboiz11/Day21-ThanTienDat-2A202603023-Track3-Finetuning-Lab.git"
PROJECT = pathlib.Path("/content/lab21_2A202603023")

if not (PROJECT / "requirements.txt").exists():
    if SOURCE == "upload":
        from google.colab import files
        print("Chọn COLAB_INPUT_2A202603023.zip (hoặc ZIP kết quả để tiếp tục).")
        uploaded = files.upload()
        archives = [name for name in uploaded if name.lower().endswith(".zip")]
        if len(archives) != 1:
            raise RuntimeError("Hãy tải lên đúng một file ZIP của dự án.")
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                destination = (pathlib.Path("/content") / member.filename).resolve()
                if not destination.is_relative_to(PROJECT) or "\\\\" in member.filename:
                    raise RuntimeError("ZIP có cấu trúc không đúng; dùng gói do export_colab.py tạo.")
            archive.extractall("/content")
    else:
        subprocess.run(["git", "clone", REPO, str(PROJECT)], check=True)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
for name in ("results", "adapters"):
    (PROJECT / name).mkdir(exist_ok=True)
required = ["data/train_seed.jsonl", "data/eval_target.jsonl", "data/eval_regression.jsonl", "data/checksums.json"]
missing = [name for name in required if not (PROJECT / name).is_file()]
if missing:
    raise RuntimeError(f"Gói ZIP thiếu dữ liệu: {missing}")

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
os.environ["COMPUTE_TIER"] = "T4"
os.environ["EPOCHS"] = "2"
os.environ["MASK_MODE"] = "assistant-only"
for name in ("EVAL_LIMIT", "BASE_MODEL", "ONLY", "FORCE_RETRAIN", "LAB_BACKUP_DIR"):
    os.environ.pop(name, None)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Chọn Runtime > Change runtime type > T4 GPU, rồi chạy lại ô Setup.")
print("GPU:", torch.cuda.get_device_name(0))
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["LAB_BACKUP_DIR"] = "/content/drive/MyDrive/Lab21_2A202603023"
    print("Sao lưu sau mỗi NB và mỗi adapter vào", os.environ["LAB_BACKUP_DIR"])

def run_stages(*stages):
    subprocess.run([sys.executable, "-u", "scripts/colab_run.py", *stages], check=True)
'''

RUN_ALL_CELLS = [
    ("markdown", """# Lab 21 — Thân Tiến Đạt · 2A202603023

Chọn **Runtime → Change runtime type → T4 GPU**. Chạy các ô **1 → 6** theo thứ tự.
Ô 1: upload `COLAB_INPUT_2A202603023.zip`; dữ liệu đã nằm trong ZIP.
Ô 3: đọc điểm baseline trước khi huấn luyện ở ô 4. Core NB1–NB5 khoảng 100–130 phút theo số đo trong repo.
`SAVE_TO_DRIVE=True` lưu tiến độ vào Drive; có thể tắt nếu chỉ muốn tải ZIP về máy.
NB6 là tuỳ chọn, dùng notebook `Lab21_06_merge_and_serve.ipynb` sau core.
"""),
    ("code", BOOTSTRAP),
    ("code", '''# @title 2. Kiểm tra môi trường, dữ liệu và unit test
subprocess.run([sys.executable, "scripts/verify.py", "--smoke"], check=True)
import json, importlib.metadata, datetime
versions = {}
for package in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes", "torchao"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "not installed"
environment = {"utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "gpu": torch.cuda.get_device_name(0),
               "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
               "python": sys.version, "versions": versions,
               "tier": os.environ["COMPUTE_TIER"], "epochs": os.environ["EPOCHS"],
               "mask_mode": os.environ["MASK_MODE"], "eval_limit": None}
(PROJECT / "results/environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
(PROJECT / "results/requirements-lock.txt").write_text(lock, encoding="utf-8")
print(json.dumps(environment, ensure_ascii=False, indent=2))
'''),
    ("code", '''# @title 3. NB1 + NB2 — chứng minh mask và đóng băng baseline
if (PROJECT / "adapters/correct/adapter_model.safetensors").exists():
    print("Đã có adapter: giữ nguyên baseline; không đo lại sau khi đã train.")
else:
    run_stages("nb1", "nb2")
frozen = json.loads((PROJECT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
print("(a) =", frozen["baseline_a"]["target"], "· (b) =", frozen["baseline_b"]["target"])
if frozen["baseline_b"]["target"] <= frozen["baseline_a"]["target"]:
    raise RuntimeError("Dừng trước khi train: prompt (b) chưa thắng (a). Cải thiện prompt và đo lại NB2 trước NB3.")
print("Đọc mốc (b) này trước khi chạy ô 4. Không sửa eval hoặc prompt (b) sau khi train.")
'''),
    ("code", '''# @title 4. NB3 + NB4 — train đúng và ba đối chứng
frozen = json.loads((PROJECT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
if frozen.get("smoke_mode") or frozen["baseline_b"]["target"] <= frozen["baseline_a"]["target"]:
    raise RuntimeError("Hoàn thành baseline toàn bộ và cải thiện (b) ở ô 3 trước khi train.")
if (PROJECT / "adapters/correct/adapter_model.safetensors").exists():
    print("Giữ adapter correct đã hoàn thành.")
else:
    run_stages("nb3")
run_stages("nb4")  # NB4 tự bỏ qua các adapter đã lưu
'''),
    ("code", '''# @title 5. NB5 — đánh giá và sinh bảng số liệu báo cáo
run_stages("nb5")
subprocess.run([sys.executable, "scripts/build_measured_report.py"], check=True)
from IPython.display import Markdown, display
display(Markdown((PROJECT / "submission/MEASURED_RESULTS.md").read_text(encoding="utf-8")))
from labkit import report
report.backup_progress(PROJECT)
'''),
    ("code", '''# @title 6. Kiểm tra và tải ZIP kết quả (cũng chạy được sau khi một ô bị lỗi)
from google.colab import files
verification = subprocess.run([sys.executable, "scripts/verify.py"])
if verification.returncode:
    print("Chưa sẵn sàng nộp. Báo cáo cần được viết từ kết quả thật; xem các FAIL ở trên.")
output = pathlib.Path("/content/LAB21_RESULTS_2A202603023.zip")
subprocess.run([sys.executable, "scripts/export_colab.py", "--mode", "progress", "--output", str(output)], check=True)
files.download(str(output))
print("Gửi ZIP kết quả để hoàn thiện phân tích REPORT.md. NB6 là tuỳ chọn.")
'''),
]


def _stamp_cell_ids(raw: dict) -> None:
    """Give every cell an id derived from its position and content.

    nbformat mints a RANDOM id per cell, so regenerating unchanged notebooks produced a
    ~90-line diff of nothing but id churn. That is not cosmetic: a diff that is always
    noise is a diff nobody reads, which is how the stale-bootstrap bug (F-18) survived
    a review. Now `make colab` on unchanged sources is a genuinely empty diff, and any
    line that does move is a line that means something.
    """
    for i, cell in enumerate(raw.get("cells", [])):
        body = "".join(cell.get("source", []))
        cell["id"] = hashlib.sha1(f"{i}\x00{body}".encode()).hexdigest()[:8]


def _cell(kind: str, body: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def _percent_cells(text: str) -> list[dict]:
    """Read the simple code/markdown percent format used by this repository."""
    cells, kind, body = [], None, []
    def finish():
        if kind is not None:
            content = "\n".join(body).strip("\n") + "\n"
            cells.append(_cell(kind, content))
    for line in text.splitlines():
        if re.match(r"^# %%($| )", line):
            finish()
            kind = "markdown" if "[markdown]" in line else "code"
            body = []
        elif kind == "markdown":
            body.append(line[2:] if line.startswith("# ") else "" if line == "#" else line)
        elif kind == "code":
            body.append(line)
    finish()
    return cells


def _write(cells: list[dict], dest: pathlib.Path) -> None:
    raw = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "colab": {"name": dest.name}, "accelerator": "GPU"},
           "nbformat": 4, "nbformat_minor": 5}
    _stamp_cell_ids(raw)
    dest.write_text(json.dumps(raw, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")


def main() -> int:

    OUT.mkdir(exist_ok=True)
    made = []
    for src in sorted(SRC.glob("*.py")):
        dest = OUT / f"Lab21_{src.stem}.ipynb"
        _write([_cell("code", BOOTSTRAP), *_percent_cells(src.read_text(encoding="utf-8"))], dest)
        made.append(dest.name)
    dest = OUT / "Lab21_RUN_ALL.ipynb"
    _write([_cell(kind, body) for kind, body in RUN_ALL_CELLS], dest)
    made.append(dest.name)
    print(f"wrote {len(made)} notebooks to colab/:")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
