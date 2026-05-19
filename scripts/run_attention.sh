#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=0

echo "[Assets] Download/cache Flowers102 and ImageNet weights if needed"
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18

echo "[Attention] Baseline ResNet-18"
python scripts/train.py --config configs/baseline_resnet18.yaml --experiment.name attention_baseline_resnet18 --attention none

echo "[Attention] SE-ResNet-18"
python scripts/train.py --config configs/se_resnet18.yaml --experiment.name attention_se_resnet18 --attention se

echo "[Attention] Optional CBAM-ResNet-18"
python scripts/train.py --config configs/baseline_resnet18.yaml --experiment.name attention_cbam_resnet18 --attention cbam

python scripts/summarize_results.py --out_csv outputs/metrics/attention_comparison.csv --out_md outputs/metrics/attention_comparison.md
python scripts/export_experiment_settings.py
