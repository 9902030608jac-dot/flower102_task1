from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from .attention import SEBlock
from .resnet_baseline import build_resnet_baseline


def _forward_with_attention(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.attention(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out


def _attach_se_to_basic_blocks(model: nn.Module, reduction: int) -> None:
    for module in model.modules():
        if isinstance(module, BasicBlock):
            channels = module.bn2.num_features
            module.attention = SEBlock(channels, reduction)
            module.forward = MethodType(_forward_with_attention, module)


def build_se_resnet(arch: str, num_classes: int = 102, pretrained: bool = True, reduction: int = 16) -> nn.Module:
    """Build ResNet with SE inserted inside every BasicBlock before residual add."""
    model = build_resnet_baseline(arch, num_classes=num_classes, pretrained=pretrained)
    _attach_se_to_basic_blocks(model, reduction)
    return model
