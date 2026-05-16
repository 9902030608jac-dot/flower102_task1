#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_normalized_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums >= 0)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks[::5])
    ax.set_yticks(ticks[::5])
    ax.set_xticklabels(ticks[::5], rotation=90, fontsize=7)
    ax.set_yticklabels(ticks[::5], fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze_one(cm_path: Path, top_k: int) -> None:
    cm = np.load(cm_path)
    out_dir = cm_path.parent
    exp_name = out_dir.name

    save_normalized_confusion(
        cm,
        out_dir / "normalized_confusion_matrix.png",
        f"{exp_name} Normalized Confusion Matrix",
    )

    pairs: list[tuple[int, int, int]] = []
    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            count = int(cm[true_id, pred_id])
            if true_id != pred_id and count > 0:
                pairs.append((count, true_id, pred_id))
    pairs.sort(reverse=True)

    with (out_dir / "top_confusion_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "true_class", "predicted_class", "error_count"])
        for rank, (count, true_id, pred_id) in enumerate(pairs[:top_k], start=1):
            writer.writerow([rank, true_id, pred_id, count])

    row_sums = cm.sum(axis=1)
    diag = np.diag(cm)
    with (out_dir / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "test_samples", "correct", "accuracy"])
        for class_id, (total, correct) in enumerate(zip(row_sums, diag)):
            acc = float(correct / total) if total else 0.0
            writer.writerow([class_id, int(total), int(correct), acc])

    print(f"[OK] {exp_name}: wrote normalized_confusion_matrix.png, top_confusion_pairs.csv, per_class_accuracy.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved Flowers102 confusion matrices.")
    parser.add_argument("--figures_root", default="outputs/figures")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.figures_root)
    cm_paths = sorted(root.glob("*/confusion_matrix.npy"))
    if not cm_paths:
        raise FileNotFoundError(f"No confusion_matrix.npy files found under {root}")
    for cm_path in cm_paths:
        analyze_one(cm_path, args.top_k)


if __name__ == "__main__":
    main()
