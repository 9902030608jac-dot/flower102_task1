#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=0

echo "[Assets] Download/cache Flowers102 and ImageNet weights if needed"
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34

echo "[Baseline] ResNet-18 pretrained"
python scripts/train.py --config configs/baseline_resnet18.yaml

echo "[Baseline] ResNet-34 pretrained"
python scripts/train.py --config configs/baseline_resnet34.yaml

python scripts/summarize_results.py
python scripts/export_experiment_settings.py
