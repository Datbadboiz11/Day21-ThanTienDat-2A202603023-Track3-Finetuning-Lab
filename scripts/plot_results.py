#!/usr/bin/env python3
"""Render report figures from recorded measurements; does not run a model."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ROOT / "submission" / "figures"
    output.mkdir(exist_ok=True)
    comparison = json.loads((ROOT / "results/verdict.json").read_text(encoding="utf-8"))["comparison"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#8494a7", "#2378b3", "#b45309"]
    labels = ["(a) Base + naive", "(b) Base + optimized", "(c) LoRA correct"]
    for ax, key, title in zip(axes, ["target", "regression"],
                              ["Target: mean field accuracy", "Regression: keyword recall"]):
        values = [row[key] * 100 for row in comparison]
        ax.bar(range(3), values, color=colors, width=0.58)
        ax.set_xticks(range(3), labels, rotation=12, ha="right")
        ax.set_ylim(0, 108)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(i, value + 2, f"{value:.2f}%", ha="center", weight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    threshold = comparison[1]["regression"] * 100 - 2
    axes[1].axhline(threshold, color="#a43131", linestyle="--", linewidth=1.2,
                    label=f"Gate minimum: {threshold:.2f}%")
    axes[1].legend(loc="lower left", fontsize=9)
    fig.suptitle("Lab 21 | Target improves, regression gate fails", fontsize=13, weight="bold")
    fig.savefig(output / "baseline_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    palette = {"correct": "#2378b3", "attn_only": "#388559",
               "wrong_lr": "#a43131", "qlora": "#b45309"}
    for key, color in palette.items():
        history = json.loads((ROOT / "results" / f"training_log_{key}.json").read_text(encoding="utf-8"))
        logs = [row for row in history if "loss" in row]
        ax.plot([row["step"] for row in logs], [row["loss"] for row in logs],
                marker="o", linewidth=2, color=color, label=key)
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Recorded trainer step")
    ax.set_ylabel("Training loss (log scale; interval averages)")
    ax.set_title("Same 30-step budget | Recorded learning curves", weight="bold")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)
    print("Saved submission/figures/baseline_comparison.png and training_loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
