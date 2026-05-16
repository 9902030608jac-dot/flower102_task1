from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .metrics import AverageMeter, accuracy, save_confusion_matrix, save_training_curves
from .utils import save_json


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    losses, accs = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        top1 = accuracy(outputs.detach(), targets, topk=(1,))[0].item()
        bs = images.size(0)
        losses.update(loss.item(), bs)
        accs.update(top1, bs)
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg:.2f}")

    return {"train_loss": losses.avg, "train_acc": accs.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
    amp: bool = True,
    return_preds: bool = True,
) -> dict[str, Any]:
    model.eval()
    losses, accs = AverageMeter(), AverageMeter()
    y_true: list[int] = []
    y_pred: list[int] = []
    pbar = tqdm(loader, desc=f"Eval {split}", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        top1 = accuracy(outputs, targets, topk=(1,))[0].item()
        bs = images.size(0)
        losses.update(loss.item(), bs)
        accs.update(top1, bs)
        if return_preds:
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg:.2f}")

    return {f"{split}_loss": losses.avg, f"{split}_acc": accs.avg, "y_true": y_true, "y_pred": y_pred}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": config,
        },
        path,
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, Any],
    paths: dict[str, Path],
    exp_logger=None,
) -> dict[str, Any]:
    train_cfg = config["train"]
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)
    epochs = int(train_cfg["epochs"])
    grad_clip = train_cfg.get("grad_clip", 1.0)
    metrics_path = paths["metrics_dir"] / "metrics.csv"
    best_ckpt = paths["ckpt_dir"] / "best.pt"
    last_ckpt = paths["ckpt_dir"] / "last.pt"
    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, amp=amp, grad_clip=grad_clip, scaler=scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device, split="val", amp=amp, return_preds=False)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        curves_path = save_training_curves(history, paths["figures_dir"])
        if exp_logger is not None:
            exp_logger.log(row, step=epoch)
            exp_logger.log_artifact(curves_path, name="training_curves")
            if hasattr(exp_logger, "save_file"):
                exp_logger.save_file(metrics_path)
                exp_logger.save_file(curves_path)

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_val_acc, config)
            if exp_logger is not None and hasattr(exp_logger, "save_file"):
                exp_logger.save_file(best_ckpt)

        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_val_acc, config)
        if exp_logger is not None and hasattr(exp_logger, "save_file"):
            exp_logger.save_file(last_ckpt)
        print(f"Epoch {epoch:03d}/{epochs}: train_acc={row['train_acc']:.2f}, val_acc={row['val_acc']:.2f}, best={best_val_acc:.2f}")

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, split="test", amp=amp, return_preds=True)
    elapsed = time.time() - start
    cm_paths = save_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"], paths["figures_dir"], num_classes=config["model"]["num_classes"]
    )
    out = {
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_metrics["test_loss"]),
        "test_acc": float(test_metrics["test_acc"]),
        "train_time": elapsed,
        "checkpoint_path": str(best_ckpt),
        "confusion_matrix_npy": cm_paths["npy"],
        "confusion_matrix_png": cm_paths["png"],
    }
    test_metrics_path = paths["metrics_dir"] / "test_metrics.json"
    save_json(out, test_metrics_path)
    if exp_logger is not None:
        exp_logger.log({"best_val_acc": best_val_acc, "test_acc": out["test_acc"], "train_time": elapsed})
        exp_logger.log_artifact(cm_paths["png"], name="confusion_matrix")
        if hasattr(exp_logger, "save_file"):
            exp_logger.save_file(test_metrics_path)
            exp_logger.save_file(cm_paths["npy"])
            exp_logger.save_file(cm_paths["png"])
    return out
