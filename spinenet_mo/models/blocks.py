"""Building blocks: Ghost Convolution, Efficient Channel Attention, CBAM and factories."""
from __future__ import annotations

import torch
import torch.nn as nn


class GhostConv(nn.Module):
    """Ghost Convolution (Han et al., 2020) -- Eqs. (5)-(6) of the manuscript.

    ``m = n / s`` intrinsic maps come from a standard convolution; the remaining ``n - m`` "ghost"
    maps are produced by cheap depth-wise transformations of the intrinsic maps; both are concatenated.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, ratio: int = 2, dw_k: int = 3):
        super().__init__()
        if cout % ratio:
            raise ValueError("cout must be divisible by the ghost ratio")
        m = cout // ratio
        self.cout = cout
        self.primary = nn.Conv2d(cin, m, k, stride, k // 2, bias=False)
        self.cheap = nn.Conv2d(m, m * (ratio - 1), dw_k, 1, dw_k // 2, groups=m, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        return torch.cat([y, self.cheap(y)], dim=1)[:, : self.cout]


class DepthwiseSeparableConv(nn.Module):
    """k x k depth-wise convolution followed by a 1 x 1 point-wise convolution."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020) -- Eqs. (7)-(8).

    The channel descriptor ``z`` (global average pooling) is passed through a 1-D convolution of
    adaptive kernel size ``k`` *without dimensionality reduction*; the sigmoid weights ``w`` are exposed
    (``last_weights``) so that channel importance can be inspected after a forward pass.
    """

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=(2, 3))                                   # (B, C)
        w = torch.sigmoid(self.conv(z.unsqueeze(1))).squeeze(1)  # (B, C)
        self.last_weights = w.detach()
        return x * w[:, :, None, None]


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_k: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, spatial_k, padding=spatial_k // 2, bias=True)

    def forward(self, x):
        avg, mx = x.mean(dim=(2, 3)), x.amax(dim=(2, 3))
        x = x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))[:, :, None, None]
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


def make_activation(kind: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(inplace=True),
    }[kind]


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "layer":          # channel-wise affine LayerNorm over (C, H, W) == GroupNorm with one group
        return nn.GroupNorm(1, channels)
    if kind == "group":
        return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
    raise ValueError(kind)
