"""PyTorch realisation of an architecture specification (see :mod:`spinenet_mo.search.arch`)."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..search.arch import build_spec
from .blocks import CBAM, ECA, DepthwiseSeparableConv, GhostConv, make_activation, make_norm


def _make_layer(layer: Dict[str, Any]) -> nn.Module:
    op = layer["op"]
    if op == "conv":
        return nn.Conv2d(layer["cin"], layer["cout"], layer["k"], layer["stride"], layer["k"] // 2, bias=False)
    if op == "dwsep":
        return DepthwiseSeparableConv(layer["cin"], layer["cout"], layer["k"], layer["stride"])
    if op == "ghost":
        return GhostConv(layer["cin"], layer["cout"], layer["k"], layer["stride"], layer["ratio"], layer["dw_k"])
    if op == "norm":
        return make_norm(layer["kind"], layer["c"])
    if op == "act":
        return make_activation(layer["kind"])
    if op == "eca":
        return ECA(layer["c"], layer["k"])
    if op == "cbam":
        return CBAM(layer["c"], layer["reduction"], layer["spatial_k"])
    if op == "gap":
        return nn.AdaptiveAvgPool2d(1)
    if op == "adaptive_pool":
        cls = nn.AdaptiveMaxPool2d if layer["kind"] == "max" else nn.AdaptiveAvgPool2d
        return cls(layer["size"])
    if op == "flatten":
        return nn.Flatten()
    if op == "dropout":
        return nn.Dropout(layer["p"])
    if op == "linear":
        return nn.Linear(layer["in"], layer["out"])
    raise ValueError(f"Unknown op {op}")


class ResidualBlock(nn.Module):
    def __init__(self, layers: Sequence[Dict[str, Any]], residual):
        super().__init__()
        self.body = nn.Sequential(*[_make_layer(l) for l in layers])
        self.shortcut: nn.Module | None = None
        self.use_residual = residual is not None
        if residual is not None and residual["project"]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(residual["cin"], residual["cout"], 1, residual["stride"], bias=False),
                nn.BatchNorm2d(residual["cout"]),
            )

    def forward(self, x):
        out = self.body(x)
        if self.use_residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))
        return out


class SpineNet(nn.Module):
    """Network decoded from a genome / specification.

    ``forward_features`` returns the last feature map (used by Grad-CAM), ``forward`` the logits.
    """

    def __init__(self, spec: Dict[str, Any]):
        super().__init__()
        self.spec = spec
        self.stem = nn.Sequential(*[_make_layer(l) for l in spec["stem"]])
        self.blocks = nn.ModuleList([ResidualBlock(b["layers"], b["residual"]) for b in spec["blocks"]])
        self.head = nn.Sequential(*[_make_layer(l) for l in spec["head"]])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def eca_weights(self):
        """Channel-attention weights of every ECA module from the last forward pass."""
        return [m.last_weights for m in self.modules() if isinstance(m, ECA) and m.last_weights is not None]


def build_model(genome: Sequence[int], num_blocks: int) -> SpineNet:
    return SpineNet(build_spec(genome, num_blocks))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
