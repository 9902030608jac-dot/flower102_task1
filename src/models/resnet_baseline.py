from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights


def build_resnet_baseline(arch: str, num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet arch: {arch}. Choose resnet18 or resnet34.")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def split_backbone_head_params(model: nn.Module):
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params
