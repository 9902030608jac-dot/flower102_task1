from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SEBlock
from .resnet_baseline import build_resnet_baseline


class StageSEResNet(nn.Module):
    """Torchvision ResNet with one SE block after each residual stage."""

    def __init__(self, arch: str, num_classes: int = 102, pretrained: bool = True, reduction: int = 16):
        super().__init__()
        base = build_resnet_baseline(arch, num_classes=num_classes, pretrained=pretrained)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.se1 = SEBlock(64, reduction)
        self.se2 = SEBlock(128, reduction)
        self.se3 = SEBlock(256, reduction)
        self.se4 = SEBlock(512, reduction)
        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_se_resnet(arch: str, num_classes: int = 102, pretrained: bool = True, reduction: int = 16) -> nn.Module:
    return StageSEResNet(arch=arch, num_classes=num_classes, pretrained=pretrained, reduction=reduction)
