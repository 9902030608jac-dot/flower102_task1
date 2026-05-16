#!/usr/bin/env bash
set -e

# export CUDA_VISIBLE_DEVICES=0

echo "[Assets] Download/cache Flowers102 and ImageNet weights if needed"
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34

echo "[Sweep] ResNet-18, backbone_lr=1e-4, head_lr=1e-3, epochs=30"
python scripts/train.py --config configs/baseline_resnet18.yaml --experiment.name hparam_resnet18_lr1_epochs30 --arch resnet18 --pretrained true --epochs 30 --backbone_lr 1e-4 --head_lr 1e-3

echo "[Sweep] ResNet-18, backbone_lr=5e-5, head_lr=5e-4, epochs=50"
python scripts/train.py --config configs/baseline_resnet18.yaml --experiment.name hparam_resnet18_lr2_epochs50 --arch resnet18 --pretrained true --epochs 50 --backbone_lr 5e-5 --head_lr 5e-4

echo "[Sweep] ResNet-34, backbone_lr=1e-4, head_lr=1e-3, epochs=30"
python scripts/train.py --config configs/baseline_resnet34.yaml --experiment.name hparam_resnet34_lr1_epochs30 --arch resnet34 --pretrained true --epochs 30 --backbone_lr 1e-4 --head_lr 1e-3

echo "[Sweep] ResNet-34, backbone_lr=5e-5, head_lr=5e-4, epochs=50"
python scripts/train.py --config configs/baseline_resnet34.yaml --experiment.name hparam_resnet34_lr2_epochs50 --arch resnet34 --pretrained true --epochs 50 --backbone_lr 5e-5 --head_lr 5e-4

python scripts/summarize_results.py --out_csv outputs/metrics/results_summary.csv --out_md outputs/metrics/results_summary.md
