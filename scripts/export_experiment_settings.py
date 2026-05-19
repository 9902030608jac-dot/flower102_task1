#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.fillna("").astype(str).iterrows():
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export detailed Flowers102 experiment settings.")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--out_csv", default="outputs/metrics/experiment_settings.csv")
    parser.add_argument("--out_md", default="outputs/metrics/experiment_settings.md")
    args = parser.parse_args()

    rows = []
    for cfg_path in sorted(Path(args.metrics_dir).glob("*/config.yaml")):
        cfg = read_yaml(cfg_path)
        exp = cfg_path.parent.name
        metrics_csv = cfg_path.parent / "metrics.csv"
        if metrics_csv.exists():
            metrics = pd.read_csv(metrics_csv)
            epochs_done = len(metrics)
        else:
            metrics = pd.DataFrame()
            epochs_done = cfg["train"].get("epochs")
        batch_size = int(cfg["data"].get("batch_size", 32))
        train_size, val_size, test_size = 1020, 1020, 6149
        train_batches = (train_size + batch_size - 1) // batch_size
        rows.append(
            {
                "experiment_name": exp,
                "dataset_split": "official Flowers102 train/val/test",
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "network": cfg["model"].get("arch"),
                "pretrained": cfg["model"].get("pretrained", True),
                "attention": cfg["model"].get("attention", "none"),
                "attention_position": "inside each BasicBlock after bn2 and before residual add" if cfg["model"].get("attention", "none") != "none" else "none",
                "batch_size": batch_size,
                "backbone_lr": cfg["train"].get("backbone_lr"),
                "attention_lr": cfg["train"].get("attention_lr"),
                "head_lr": cfg["train"].get("head_lr"),
                "optimizer": cfg["train"].get("optimizer"),
                "scheduler": cfg["train"].get("scheduler"),
                "epochs_configured": cfg["train"].get("epochs"),
                "epochs_completed": epochs_done,
                "train_batches_per_epoch": train_batches,
                "total_train_iterations": train_batches * int(cfg["train"].get("epochs", epochs_done)),
                "loss_function": "CrossEntropyLoss",
                "evaluation_metrics": "Top-1 Accuracy, macro mAP",
                "augmentation": "RandomResizedCrop, RandomHorizontalFlip, light ColorJitter, ImageNet Normalize",
                "eval_transform": "Resize, CenterCrop, ImageNet Normalize",
                "logger": cfg.get("logger", {}).get("type", "none"),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_md.write_text(to_markdown(df), encoding="utf-8")
    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
