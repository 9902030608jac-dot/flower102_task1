#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import build_dataloaders
from src.engine import fit
from src.logger import ExperimentLogger
from src.models import build_cbam_resnet, build_resnet_baseline, build_se_resnet
from src.models.resnet_baseline import split_backbone_head_params
from src.utils import count_parameters, create_dirs, get_device, load_yaml, parse_value, save_json, save_yaml, set_by_dotted_key, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet variants on Flowers102.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--pretrained", type=str, choices=["true", "false", "True", "False"])
    parser.add_argument("--arch", type=str, choices=["resnet18", "resnet34"])
    parser.add_argument("--attention", type=str, choices=["none", "se", "cbam"])
    parser.add_argument("--logger", type=str, choices=["none", "swanlab"])
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


def apply_overrides(config, args):
    explicit = {
        "train.epochs": args.epochs,
        "data.batch_size": args.batch_size,
        "train.backbone_lr": args.backbone_lr,
        "train.head_lr": args.head_lr,
        "train.weight_decay": args.weight_decay,
        "model.arch": args.arch,
        "model.attention": args.attention,
        "logger.type": args.logger,
    }
    if args.pretrained is not None:
        explicit["model.pretrained"] = args.pretrained.lower() == "true"
    for key, value in explicit.items():
        if value is not None:
            set_by_dotted_key(config, key, value)

    tokens = args.overrides
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if key.startswith("--") and i + 1 < len(tokens):
            set_by_dotted_key(config, key[2:], parse_value(tokens[i + 1]))
            i += 2
        else:
            i += 1
    return config


def build_model(config):
    m = config["model"]
    attention = str(m.get("attention", "none")).lower()
    if attention == "none":
        return build_resnet_baseline(m["arch"], m["num_classes"], m.get("pretrained", True))
    if attention == "se":
        return build_se_resnet(m["arch"], m["num_classes"], m.get("pretrained", True), m.get("reduction", 16))
    if attention == "cbam":
        return build_cbam_resnet(m["arch"], m["num_classes"], m.get("pretrained", True), m.get("reduction", 16))
    raise ValueError(f"Unknown attention: {attention}")


def build_optimizer(config, model):
    train_cfg = config["train"]
    backbone_params, head_params = split_backbone_head_params(model)
    groups = [
        {"params": backbone_params, "lr": float(train_cfg["backbone_lr"]), "name": "backbone"},
        {"params": head_params, "lr": float(train_cfg["head_lr"]), "name": "head"},
    ]
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    wd = float(train_cfg.get("weight_decay", 1e-4))
    if opt_name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(config, optimizer):
    scheduler_name = str(config["train"].get("scheduler", "cosine")).lower()
    epochs = int(config["train"]["epochs"])
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name in {"none", "null"}:
        return None
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main():
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args)
    torch_home = config.get("assets", {}).get("torch_home")
    if torch_home:
        torch_home_path = (ROOT / torch_home).resolve()
        os.environ["TORCH_HOME"] = str(torch_home_path)
        torch.hub.set_dir(str(torch_home_path / "hub"))
    set_seed(int(config["train"].get("seed", 42)))
    exp_name = config["experiment"]["name"]
    output_dir = ROOT / config["experiment"].get("output_dir", "outputs")
    paths = {
        "ckpt_dir": output_dir / "checkpoints" / exp_name,
        "metrics_dir": output_dir / "metrics" / exp_name,
        "figures_dir": output_dir / "figures" / exp_name,
        "logs_dir": output_dir / "logs" / exp_name,
    }
    create_dirs(*paths.values())
    save_yaml(config, paths["metrics_dir"] / "config.yaml")

    device = get_device()
    train_loader, val_loader, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss()
    params = count_parameters(model)

    print("========== Experiment ==========")
    print(f"name: {exp_name}")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}, torchvision: {torchvision.__version__}")
    print(f"model: {config['model']['arch']}, pretrained: {config['model']['pretrained']}, attention: {config['model'].get('attention', 'none')}")
    print(f"sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"batch_size: {config['data']['batch_size']}, optimizer: {config['train']['optimizer']}")
    print(f"lr: backbone={config['train']['backbone_lr']}, head={config['train']['head_lr']}, wd={config['train']['weight_decay']}")
    print(f"params: {params:,}")

    exp_logger = ExperimentLogger(logger_type=config["logger"].get("type", "none"), config=config)
    if hasattr(exp_logger, "save_file"):
        exp_logger.save_file(paths["metrics_dir"] / "config.yaml")
    summary = fit(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, device, config, paths, exp_logger)
    summary.update(
        {
            "experiment_name": exp_name,
            "model": config["model"]["arch"],
            "pretrained": config["model"]["pretrained"],
            "attention": config["model"].get("attention", "none"),
            "epochs": config["train"]["epochs"],
            "batch_size": config["data"]["batch_size"],
            "backbone_lr": config["train"]["backbone_lr"],
            "head_lr": config["train"]["head_lr"],
            "weight_decay": config["train"]["weight_decay"],
            "params": params,
        }
    )
    summary_json_path = paths["metrics_dir"] / "summary.json"
    summary_csv_path = paths["metrics_dir"] / "summary.csv"
    save_json(summary, summary_json_path)
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    if hasattr(exp_logger, "save_file"):
        exp_logger.save_file(summary_json_path)
        exp_logger.save_file(summary_csv_path)
    exp_logger.finish()


if __name__ == "__main__":
    main()
