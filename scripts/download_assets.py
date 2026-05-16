#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision.datasets import Flowers102
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from src.transforms import build_eval_transform, build_train_transform
from src.utils import create_dirs, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Download Flowers102 dataset and torchvision ImageNet weights.")
    parser.add_argument("--config", default="configs/baseline_resnet18.yaml", help="Config used to read data.root and image_size.")
    parser.add_argument("--data_root", default=None, help="Override dataset root. Default: config data.root.")
    parser.add_argument("--torch_home", default=None, help="Override torch cache root. Default: config assets.torch_home or assets/torch.")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet34"], choices=["resnet18", "resnet34"])
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_models", action="store_true")
    return parser.parse_args()


def download_flowers102(data_root: Path, image_size: int) -> None:
    print(f"[data] root: {data_root.resolve()}")
    create_dirs(data_root)
    train_t = build_train_transform(image_size=image_size)
    eval_t = build_eval_transform(image_size=image_size)
    for split, transform in [("train", train_t), ("val", eval_t), ("test", eval_t)]:
        dataset = Flowers102(root=str(data_root), split=split, download=True, transform=transform)
        print(f"[data] Flowers102 split={split}, size={len(dataset)}")
    print(f"[data] torchvision stores files under: {(data_root / 'flowers-102').resolve()}")


def download_models(torch_home: Path, models: list[str]) -> None:
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    torch.hub.set_dir(str((torch_home / "hub").resolve()))
    create_dirs(torch_home / "hub" / "checkpoints")
    print(f"[model] TORCH_HOME: {torch_home.resolve()}")
    print(f"[model] torch hub dir: {Path(torch.hub.get_dir()).resolve()}")
    print(f"[model] checkpoint dir: {(Path(torch.hub.get_dir()) / 'checkpoints').resolve()}")

    for name in models:
        if name == "resnet18":
            _ = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif name == "resnet34":
            _ = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        print(f"[model] cached ImageNet weights for {name}")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    data_root = Path(args.data_root or config.get("data", {}).get("root", "data"))
    torch_home = Path(args.torch_home or config.get("assets", {}).get("torch_home", "assets/torch"))
    image_size = int(config.get("data", {}).get("image_size", 224))

    if not args.skip_data:
        download_flowers102(data_root, image_size)
    if not args.skip_models:
        download_models(torch_home, args.models)
    print("[done] assets are ready.")


if __name__ == "__main__":
    main()
