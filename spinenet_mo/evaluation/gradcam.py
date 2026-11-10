"""Grad-CAM for the last feature map of a SpineNet network (qualitative interpretability, Fig. 10)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on ``model.forward_features`` output."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x: torch.Tensor, target: Optional[int] = None) -> np.ndarray:
        """Return heat-maps ``(B, H, W)`` in [0, 1] upsampled to the input size."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        feats = self.model.forward_features(x)
        feats.retain_grad()
        logits = self.model.head(feats)
        cls = logits.argmax(dim=1) if target is None else torch.full((x.shape[0],), target, device=x.device)
        score = logits.gather(1, cls[:, None]).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)                    # (B, C, 1, 1)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))               # (B, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
        mins = cam.min(axis=(1, 2), keepdims=True)
        maxs = cam.max(axis=(1, 2), keepdims=True)
        return (cam - mins) / np.maximum(maxs - mins, 1e-8)
