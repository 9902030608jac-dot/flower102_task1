#!/usr/bin/env bash
set -e

# Optional fairness-oriented experiments. This script is not part of the
# minimum report pipeline; run it only when extra training budget is available.
# It gives scratch and attention models a larger optimization budget before
# drawing stronger conclusions about their final performance.

# export CUDA_VISIBLE_DEVICES=0

python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34

# Scratch models start from random initialization. Larger backbone LR and more
# epochs are usually needed than in ImageNet-pretrained fine-tuning.
python scripts/train.py \
  --config configs/scratch_resnet18.yaml \
  --experiment.name extended_scratch_resnet18_epochs100_lr1e3 \
  --arch resnet18 \
  --pretrained false \
  --attention none \
  --epochs 100 \
  --batch_size 64 \
  --backbone_lr 1e-3 \
  --head_lr 1e-3 \
  --logger none

python scripts/train.py \
  --config configs/baseline_resnet34.yaml \
  --experiment.name extended_scratch_resnet34_epochs100_lr1e3 \
  --arch resnet34 \
  --pretrained false \
  --attention none \
  --epochs 100 \
  --batch_size 32 \
  --backbone_lr 1e-3 \
  --head_lr 1e-3 \
  --logger none

# Attention modules are newly inserted and randomly initialized on top of the
# pretrained ResNet backbone. A longer run helps distinguish optimization
# difficulty from the intrinsic value of SE/CBAM.
python scripts/train.py \
  --config configs/se_resnet18.yaml \
  --experiment.name extended_attention_se_resnet18_epochs50 \
  --attention se \
  --pretrained true \
  --epochs 50 \
  --batch_size 64 \
  --backbone_lr 1e-4 \
  --head_lr 1e-3 \
  --logger none

python scripts/train.py \
  --config configs/baseline_resnet18.yaml \
  --experiment.name extended_attention_cbam_resnet18_epochs50 \
  --attention cbam \
  --pretrained true \
  --epochs 50 \
  --batch_size 64 \
  --backbone_lr 1e-4 \
  --head_lr 1e-3 \
  --logger none

python scripts/summarize_results.py \
  --out_csv outputs/metrics/extended_budget_summary.csv \
  --out_md outputs/metrics/extended_budget_summary.md

python - <<'PYFILTER'
from pathlib import Path
import pandas as pd

experiments = [
    "extended_scratch_resnet18_epochs100_lr1e3",
    "extended_scratch_resnet34_epochs100_lr1e3",
    "extended_attention_se_resnet18_epochs50",
    "extended_attention_cbam_resnet18_epochs50",
]
columns = [
    "experiment_name",
    "model",
    "pretrained",
    "attention",
    "epochs",
    "batch_size",
    "backbone_lr",
    "head_lr",
    "best_val_acc",
    "test_acc",
    "params",
]
summary = pd.read_csv("outputs/metrics/extended_budget_summary.csv")
subset = summary[summary["experiment_name"].isin(experiments)].copy()
subset = subset.sort_values(["model", "attention", "pretrained", "epochs", "experiment_name"])
subset = subset[columns]
subset.to_csv("outputs/metrics/extended_budget_only.csv", index=False)

def fmt(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

lines = [
    "| " + " | ".join(columns) + " |",
    "| " + " | ".join(["---"] * len(columns)) + " |",
]
for _, row in subset.iterrows():
    lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
Path("outputs/metrics/extended_budget_only.md").write_text("\n".join(lines) + "\n")
PYFILTER
