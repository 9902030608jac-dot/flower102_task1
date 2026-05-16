#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        cols = list(df.columns)
        return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |"
    str_df = df.fillna("").astype(str)
    cols = list(str_df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in str_df.iterrows():
        lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize all experiment metrics.")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    parser.add_argument("--out_csv", default="outputs/metrics/results_summary.csv")
    parser.add_argument("--out_md", default="outputs/metrics/results_summary.md")
    args = parser.parse_args()

    metrics_root = Path(args.metrics_dir)
    rows = []
    for exp_dir in sorted(metrics_root.iterdir() if metrics_root.exists() else []):
        if not exp_dir.is_dir():
            continue
        cfg_path = exp_dir / "config.yaml"
        test_path = exp_dir / "test_metrics.json"
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            row = read_json(summary_path)
        elif cfg_path.exists() and test_path.exists():
            cfg = read_yaml(cfg_path)
            test = read_json(test_path)
            row = {
                "experiment_name": cfg["experiment"]["name"],
                "model": cfg["model"]["arch"],
                "pretrained": cfg["model"].get("pretrained", True),
                "attention": cfg["model"].get("attention", "none"),
                "epochs": cfg["train"]["epochs"],
                "batch_size": cfg["data"]["batch_size"],
                "backbone_lr": cfg["train"]["backbone_lr"],
                "head_lr": cfg["train"]["head_lr"],
                "weight_decay": cfg["train"]["weight_decay"],
                **test,
            }
        else:
            continue
        rows.append(row)

    columns = [
        "experiment_name",
        "model",
        "pretrained",
        "attention",
        "epochs",
        "batch_size",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "best_val_acc",
        "test_acc",
        "params",
        "train_time",
        "checkpoint_path",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns].sort_values(["model", "attention", "pretrained", "experiment_name"])

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_md.write_text(dataframe_to_markdown(df), encoding="utf-8")
    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
