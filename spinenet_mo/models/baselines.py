"""Manually designed baselines: ResNet-50, VGG-16, and CNN + random-forest hybrids (VGG16+RF, ...)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class ChannelRepeat(nn.Module):
    """Repeat a single-channel image to three channels for ImageNet-style backbones."""

    def forward(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class Baseline(nn.Module):
    """Backbone + classification head with an accessible penultimate feature vector."""

    def __init__(self, backbone: nn.Module, feature_dim: int, n_classes: int, name: str):
        super().__init__()
        self.name = name
        self.repeat = ChannelRepeat()
        self.backbone = backbone                 # maps (B, 3, H, W) -> (B, feature_dim)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def features(self, x):
        return self.backbone(self.repeat(x))

    def forward(self, x):
        return self.classifier(self.features(x))


def _weights(pretrained: bool, enum):
    return enum.DEFAULT if pretrained else None


def build_baseline(name: str, n_classes: int = 3, pretrained: bool = False) -> Tuple[Baseline, int]:
    """Return ``(model, input_size)`` for ``resnet50``, ``vgg16`` or ``inceptionv3`` (with or without ``_rf``)."""
    base = name.replace("_rf", "")
    if base == "resnet50":
        net = tvm.resnet50(weights=_weights(pretrained, tvm.ResNet50_Weights))
        net.fc = nn.Identity()
        return Baseline(net, 2048, n_classes, name), 224
    if base == "vgg16":
        net = tvm.vgg16(weights=_weights(pretrained, tvm.VGG16_Weights))
        net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])   # keep the 4096-d feature layer
        return Baseline(net, 4096, n_classes, name), 224
    if base == "inceptionv3":
        net = tvm.inception_v3(weights=_weights(pretrained, tvm.Inception_V3_Weights), aux_logits=True,
                               init_weights=not pretrained)
        net.aux_logits, net.AuxLogits = False, None
        net.fc = nn.Identity()
        return Baseline(net, 2048, n_classes, name), 299
    raise ValueError(f"Unknown baseline {name}")
