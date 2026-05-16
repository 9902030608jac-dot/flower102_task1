from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

from .transforms import build_eval_transform, build_train_transform

NUM_CLASSES = 102


def build_dataloaders(config: dict[str, Any]):
    data_cfg = config["data"]
    root = Path(data_cfg.get("root", "data"))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_set = Flowers102(
        root=root,
        split="train",
        download=True,
        transform=build_train_transform(image_size, data_cfg.get("color_jitter", True)),
    )
    val_set = Flowers102(root=root, split="val", download=True, transform=build_eval_transform(image_size))
    test_set = Flowers102(root=root, split="test", download=True, transform=build_eval_transform(image_size))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader
