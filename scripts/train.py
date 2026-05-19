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
from src.utils import count_parameters, create_dirs, get_device, load_yaml, parse_value, save_json, save_yaml, set_by_dotted_key, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet variants on Flowers102.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--attention_lr", type=float)
    parser.add_argument("--head_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--pretrained", type=str, choices=["true", "false", "True", "False"])
    parser.add_argument("--arch", type=str, choices=["resnet18", "resnet34"])
    parser.add_argument("--attention", type=str, choices=["none", "se", "cbam"])
    parser.add_argument("--logger", type=str, choices=["none", "wandb", "swanlab"])
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


def apply_overrides(config, args):
    explicit = {
        "train.epochs": args.epochs,
        "data.batch_size": args.batch_size,
        "train.backbone_lr": args.backbone_lr,
        "train.attention_lr": args.attention_lr,
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
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    attention_params = [p for name, p in model.named_parameters() if ".attention." in name]
    attention_ids = {id(p) for p in attention_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and id(p) not in attention_ids]
    groups = [
        {"params": backbone_params, "lr": float(train_cfg["backbone_lr"]), "name": "backbone"},
    ]
    if attention_params:
        groups.append({"params": attention_params, "lr": float(train_cfg.get("attention_lr", train_cfg["head_lr"])), "name": "attention"})
    groups.append({"params": head_params, "lr": float(train_cfg["head_lr"]), "name": "head"})
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
    train_iterations = len(train_loader) * int(config["train"]["epochs"])

    print("========== Experiment ==========")
    print(f"name: {exp_name}")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}, torchvision: {torchvision.__version__}")
    print("model: {}, pretrained: {}, attention: {}".format(config["model"]["arch"], config["model"]["pretrained"], config["model"].get("attention", "none")))
    print(f"sizes: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print("batch_size: {}, optimizer: {}".format(config["data"]["batch_size"], config["train"]["optimizer"]))
    print("lr: backbone={}, attention={}, head={}, wd={}".format(config["train"]["backbone_lr"], config["train"].get("attention_lr"), config["train"]["head_lr"], config["train"]["weight_decay"]))
    print("epochs: {}, train_batches_per_epoch={}, total_train_iterations={}".format(config["train"]["epochs"], len(train_loader), train_iterations))
    print("loss: CrossEntropyLoss, metrics: Top-1 Accuracy and macro mAP")
    print(f"params: {params:,}")

    logger_cfg = config.get("logger", {})
    exp_logger = ExperimentLogger(
        logger_type=logger_cfg.get("type", "none"),
        project=logger_cfg.get("project", "flower102-task1"),
        name=exp_name,
        config=config,
        log_dir=paths["logs_dir"],
    )
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
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
            "total_train_iterations": train_iterations,
            "backbone_lr": config["train"]["backbone_lr"],
            "attention_lr": config["train"].get("attention_lr"),
            "head_lr": config["train"]["head_lr"],
            "optimizer": config["train"]["optimizer"],
            "scheduler": config["train"].get("scheduler"),
            "loss_function": "CrossEntropyLoss",
            "metrics": "Top-1 Accuracy, macro mAP",
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
