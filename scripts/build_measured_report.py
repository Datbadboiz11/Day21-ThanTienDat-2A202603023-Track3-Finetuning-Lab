#!/usr/bin/env python3
"""Create measured tables from real NB1-NB5 outputs; never invent missing results."""
from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import report

REQUIRED = ("mask_proof.json", "template_check.json", "token_stats.json",
            "baselines_frozen.json", "runs.csv", "verdict.json", "autopsy.json", "qualitative.json")


def table(rows: list[dict], columns: list[str]) -> str:
    def escape(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return report.markdown_table([{key: escape(row.get(key, "")) for key in columns}
                                  for row in rows], columns)


def render(root: pathlib.Path) -> str:
    missing = [name for name in REQUIRED if not (root / "results" / name).is_file()]
    if missing:
        raise FileNotFoundError("Run NB1-NB5 first. Missing: " + ", ".join(missing))
    def load(name):
        return json.loads((root / "results" / name).read_text(encoding="utf-8"))
    proof, template, stats = load("mask_proof.json"), load("template_check.json"), load("token_stats.json")
    frozen, verdict, autopsy = load("baselines_frozen.json"), load("verdict.json"), load("autopsy.json")
    qualitative = load("qualitative.json")
    with (root / "results" / "runs.csv").open(encoding="utf-8", newline="") as handle:
        latest = {row["run"]: row for row in csv.DictReader(handle)}
    missing_runs = {"correct", "attn_only", "wrong_lr", "qlora"} - latest.keys()
    missing_scores = {"correct", "attn_only", "wrong_lr", "qlora"} - {row["run"] for row in autopsy}
    if missing_runs or missing_scores:
        raise ValueError(f"Incomplete contrasts: train={sorted(missing_runs)}, eval={sorted(missing_scores)}")
    if any("baseline_b_pred" not in row or "outcome" not in row for row in qualitative):
        raise ValueError("qualitative.json is from an older NB5; use per-ticket comparisons with baseline (b)")
    scores = {row["run"]: row for row in autopsy}
    contrasts = [{**latest[key], "target": scores[key]["target"], "format": scores[key]["format"]}
                 for key in ("correct", "attn_only", "wrong_lr", "qlora")]
    losses = [row for row in qualitative if row["outcome"] == "loss"]
    wins = [row for row in qualitative if row["outcome"] == "win"]
    selected = losses[:2] + wins[-2:]
    indices = {row["i"] for row in selected}
    for row in qualitative:
        if len(selected) >= 5:
            break
        if row["i"] not in indices:
            selected.append(row)
            indices.add(row["i"])
    status = "PASSED" if verdict["verdict"]["passed"] else "FAILED"
    warnings = []
    if frozen.get("smoke_mode"):
        warnings.append("**CHƯA ĐỦ ĐIỀU KIỆN NỘP:** đang dùng EVAL_LIMIT; cần đánh giá toàn bộ dữ liệu.")
    if len(losses) < 2:
        warnings.append(f"Chỉ đo được {len(losses)} ca FT thua (b). Chưa đủ 2 ca thua theo rubric; ghi đúng thực tế, không gán ca hoà thành thua.")
    gpu_path = root / "results" / "environment.json"
    environment = json.loads(gpu_path.read_text(encoding="utf-8")) if gpu_path.exists() else {}
    return "\n\n".join([
        "# Số liệu thực nghiệm — Lab 21\n\nHọc viên: **Thân Tiến Đạt** · MSSV: **2A202603023**",
        "Tài liệu này được sinh từ kết quả thật. Phân tích và phản tư cá nhân nằm trong REPORT.md.",
        "\n\n".join(warnings),
        f"## Cấu hình đã chạy\n\nTier: `{frozen['tier']}` · Model: `{frozen['model']}` · "
        f"Target: {frozen['n_target']} mẫu · Regression: {frozen['n_regression']} mẫu.\n\n"
        f"GPU thực tế: {environment.get('gpu', 'chưa ghi nhận')}.",
        "## Bằng chứng mask và token\n\nNguồn: results/mask_proof.json, template_check.json, token_stats.json.\n\n"
        + table([{"chỉ số": key, "giá trị": value} for key, value in proof.items()
                 if not key.endswith("preview")], ["chỉ số", "giá trị"])
        + "\n\nĐoạn được tính loss:\n\n```text\n" + proof.get("supervised_preview", "") + "\n```\n\n"
        + "Template:\n\n```json\n" + json.dumps(template, ensure_ascii=False, indent=2) + "\n```\n\n"
        + "Thống kê token:\n\n```json\n" + json.dumps(stats, ensure_ascii=False, indent=2) + "\n```",
        "## So sánh ba baseline\n\nNguồn: results/verdict.json; baseline đóng băng tại NB2.\n\n"
        + table(verdict["comparison"], ["run", "target", "regression", "format", "latency_ms", "n"]),
        "## Bốn cấu hình huấn luyện\n\nNguồn: results/runs.csv và results/autopsy.json.\n\n"
        + table(contrasts, ["run", "placement", "r", "trainable_params", "learning_rate", "max_steps",
                            "final_loss", "target", "format", "train_seconds", "peak_vram_gb"])
        + "\n\nLịch sử loss: results/training_log_<run>.json. Xếp hạng bằng target, không bằng train loss.",
        f"## Cổng hồi quy: {status}\n\n```json\n" + json.dumps(verdict["verdict"], ensure_ascii=False, indent=2) + "\n```",
        f"## Ví dụ đối chiếu với (b)\n\n{len(wins)} ca thắng, {len(losses)} ca thua, "
        f"{len(qualitative)-len(wins)-len(losses)} ca hoà. Nguồn: results/qualitative.json.\n\n"
        + table(selected, ["i", "ticket", "label", "baseline_b_pred", "ft_pred", "base_score", "ft_score", "outcome"]),
        "## Đọc cùng báo cáo chính\n\nREPORT.md chứa phần giải thích đối chứng, diễn giải phán quyết, "
        "kết luận và giới hạn thí nghiệm; REFLECTION.md ghi phản tư và việc sử dụng AI. "
        "Các bảng ở đây chỉ được sinh từ số đo, không thay thế phân tích. "
        "Không thay đổi eval hay ngưỡng chấm để tạo kết quả đẹp."
    ]) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        content = render(ROOT)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Cannot build measured report: {exc}", file=sys.stderr)
        return 1
    output = ROOT / "submission" / "MEASURED_RESULTS.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output}. Complete REPORT.md using these measured tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
