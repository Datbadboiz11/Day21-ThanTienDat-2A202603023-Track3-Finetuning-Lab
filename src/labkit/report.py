"""Results I/O. Every run appends one row with the same columns, so the report table
writes itself and the grader can verify your numbers are internally consistent."""
from __future__ import annotations

import csv
import json
import pathlib
import os
import subprocess
import sys
from typing import Iterable

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"


def compare_predictions(target: list[dict], baseline: list[dict], tuned: list[str]) -> list[dict]:
    """Compare the SAME tickets, retaining full answers as evidence for the report."""
    from .evaluate import triage_field_accuracy

    if not (len(target) == len(baseline) == len(tuned)):
        raise ValueError("prediction counts differ — re-run NB2 and NB5 on the same eval set")
    rows = []
    for i, (record, base, pred) in enumerate(zip(target, baseline, tuned)):
        if base["input"] != record["input"] or base["label"] != record["label"]:
            raise ValueError(f"baseline ticket/label mismatch at {i} — do not compare different eval sets")
        b = triage_field_accuracy(base["baseline_b"], record["label"])
        c = triage_field_accuracy(pred, record["label"])
        rows.append({"i": i, "ticket": record["input"], "label": record["label"],
                     "baseline_b_pred": base["baseline_b"], "ft_pred": pred,
                     "base_score": b, "ft_score": c, "delta": c - b,
                     "outcome": "win" if c > b else "loss" if c < b else "tie"})
    return sorted(rows, key=lambda row: (row["delta"], row["i"]))


def backup_progress(root: pathlib.Path) -> None:
    """Save completed artifacts only when the user selected a backup directory."""
    destination = os.environ.get("LAB_BACKUP_DIR", "")
    if not destination:
        return
    proc = subprocess.run([sys.executable, str(root / "scripts" / "export_colab.py"),
                           "--mode", "progress", "--output",
                           str(pathlib.Path(destination) / "LAB21_PROGRESS_2A202603023.zip")])
    if proc.returncode:
        print("WARNING: backup failed; download progress using the last Colab cell.", flush=True)


def append_row(row: dict, filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Append `row`, creating the header from its keys on first write.

    Rows with new keys are unioned into the header rather than silently dropped —
    losing a column because run 3 measured something run 1 did not is exactly the kind
    of quiet data loss that makes a results table untrustworthy.
    """
    out_dir = results_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))

    rows = existing + [row]
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def write_json(obj, filename: str, results_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = results_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_rows(filename: str = "runs.csv", results_dir: pathlib.Path | None = None) -> list[dict]:
    path = (results_dir or RESULTS) / filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def markdown_table(rows: Iterable[dict], columns: list[str] | None = None) -> str:
    """Paste-ready Markdown for REPORT.md."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_"
    cols = columns or list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, rule, *body])
