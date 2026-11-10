"""Grad-CAM overlays for one correctly classified test image per class (manuscript Fig. 8)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..evaluation.gradcam import GradCAM
from ..training.trainer import get_device
from ..utils import ensure_dir
from ..viz.plots import figure_path, plot_gradcam_grid
from .context import Context
from .final import load_trained


def make_gradcam_figure(ctx: Context, checkpoint, out_dir, number: int = 8, per_class: int = 1) -> Path:
    device = get_device()
    model, _ = load_trained(checkpoint, device)
    model.to(device).eval()
    loader = ctx.loaders(ctx.cfg["train"]["batch_size"], ctx.cfg["seed"])["test"]
    cam = GradCAM(model)
    picked = {c: [] for c in range(len(ctx.class_names))}
    for x, y, _, _ in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for i in range(len(y)):
            c = int(y[i])
            if pred[i] == y[i] and len(picked[c]) < per_class:
                picked[c].append(x[i])
        if all(len(v) >= per_class for v in picked.values()):
            break
    xs = [t for c in sorted(picked) for t in picked[c]]
    titles = [ctx.class_names[c] for c in sorted(picked) for _ in picked[c]]
    batch = torch.stack(xs).to(device)
    heat = cam(batch)
    imgs = batch[:, 0].detach().cpu().numpy()
    imgs = (imgs - imgs.min(axis=(1, 2), keepdims=True)) / np.maximum(np.ptp(imgs, axis=(1, 2), keepdims=True), 1e-8)
    return plot_gradcam_grid(imgs, heat, titles, figure_path(ensure_dir(out_dir), number))
