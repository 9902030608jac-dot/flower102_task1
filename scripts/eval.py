#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import build_model
from src.datasets import build_dataloaders
from src.engine import evaluate
from src.metrics import save_confusion_matrix
from src.utils import get_device, load_yaml, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on Flowers102.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_yaml(args.config)
    torch_home = config.get("assets", {}).get("torch_home")
    if torch_home:
        torch_home_path = (ROOT / torch_home).resolve()
        os.environ["TORCH_HOME"] = str(torch_home_path)
        torch.hub.set_dir(str(torch_home_path / "hub"))
    device = get_device()
    loaders = build_dataloaders(config)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[args.split]
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device, split=args.split, amp=device.type == "cuda", return_preds=True)
    exp_name = config["experiment"]["name"]
    out_dir = ROOT / config["experiment"].get("output_dir", "outputs") / "metrics" / exp_name
    fig_dir = ROOT / config["experiment"].get("output_dir", "outputs") / "figures" / exp_name
    cm = save_confusion_matrix(metrics["y_true"], metrics["y_pred"], fig_dir, config["model"]["num_classes"], f"{args.split}_confusion_matrix")
    result = {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred"}}
    result.update({"checkpoint": args.checkpoint, **cm})
    save_json(result, out_dir / f"{args.split}_eval_metrics.json")
    print(result)


if __name__ == "__main__":
    main()
