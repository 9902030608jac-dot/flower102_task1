#!/usr/bin/env bash
set -e

# Completes a small 2x2 factorial sweep for epoch count and differential learning rate.
# Existing runs already cover lr1+30 epochs and lr2+50 epochs.

python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34

echo "[Extra Sweep] ResNet-18, backbone_lr=1e-4, head_lr=1e-3, epochs=50"
python scripts/train.py --config configs/baseline_resnet18.yaml \
  --experiment.name hparam_resnet18_lr1_epochs50 \
  --arch resnet18 --pretrained true --epochs 50 \
  --backbone_lr 1e-4 --head_lr 1e-3

echo "[Extra Sweep] ResNet-18, backbone_lr=5e-5, head_lr=5e-4, epochs=30"
python scripts/train.py --config configs/baseline_resnet18.yaml \
  --experiment.name hparam_resnet18_lr2_epochs30 \
  --arch resnet18 --pretrained true --epochs 30 \
  --backbone_lr 5e-5 --head_lr 5e-4

echo "[Extra Sweep] ResNet-34, backbone_lr=1e-4, head_lr=1e-3, epochs=50"
python scripts/train.py --config configs/baseline_resnet34.yaml \
  --experiment.name hparam_resnet34_lr1_epochs50 \
  --arch resnet34 --pretrained true --epochs 50 \
  --backbone_lr 1e-4 --head_lr 1e-3

echo "[Extra Sweep] ResNet-34, backbone_lr=5e-5, head_lr=5e-4, epochs=30"
python scripts/train.py --config configs/baseline_resnet34.yaml \
  --experiment.name hparam_resnet34_lr2_epochs30 \
  --arch resnet34 --pretrained true --epochs 30 \
  --backbone_lr 5e-5 --head_lr 5e-4

python scripts/analyze_confusions.py --top_k 10
python scripts/summarize_results.py --out_csv outputs/metrics/results_summary.csv --out_md outputs/metrics/results_summary.md
