from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_training_curves(history: list[dict], out_dir: str | Path, filename: str = "training_curves") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{filename}.png"
    if not history:
        return str(png_path)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss", marker="o", linewidth=1.8)
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val Loss", marker="o", linewidth=1.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train Acc", marker="o", linewidth=1.8)
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Val Acc", marker="o", linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    backbone_lrs = [row["lr_backbone"] for row in history]
    head_lrs = [row["lr_head"] for row in history]
    axes[2].plot(epochs, backbone_lrs, label="Backbone LR", marker="o", linewidth=1.8)
    axes[2].plot(epochs, head_lrs, label="Head LR", marker="o", linewidth=1.8)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    if all(lr > 0 for lr in backbone_lrs + head_lrs):
        axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return str(png_path)


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    out_dir: str | Path,
    num_classes: int = 102,
    filename: str = "confusion_matrix",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    npy_path = out_dir / f"{filename}.npy"
    png_path = out_dir / f"{filename}.png"
    np.save(npy_path, cm)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Flowers102 Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks[::5])
    ax.set_yticks(ticks[::5])
    ax.set_xticklabels(ticks[::5], rotation=90, fontsize=7)
    ax.set_yticklabels(ticks[::5], fontsize=7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"npy": str(npy_path), "png": str(png_path)}
