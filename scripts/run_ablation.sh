#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=0

echo "[Assets] Download/cache Flowers102 and ImageNet weights if needed"
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34

echo "[Ablation] ResNet-18 pretrained"
python scripts/train.py --config configs/baseline_resnet18.yaml --experiment.name ablation_resnet18_pretrained --pretrained true

echo "[Ablation] ResNet-18 scratch"
python scripts/train.py --config configs/scratch_resnet18.yaml --experiment.name ablation_resnet18_scratch --pretrained false

echo "[Ablation] ResNet-34 pretrained"
python scripts/train.py --config configs/baseline_resnet34.yaml --experiment.name ablation_resnet34_pretrained --pretrained true

echo "[Ablation] ResNet-34 scratch"
python scripts/train.py --config configs/baseline_resnet34.yaml --experiment.name ablation_resnet34_scratch --arch resnet34 --pretrained false

python scripts/summarize_results.py --out_csv outputs/metrics/pretrained_ablation.csv --out_md outputs/metrics/pretrained_ablation.md
python scripts/export_experiment_settings.py
