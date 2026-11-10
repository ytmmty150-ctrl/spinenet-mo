"""Figures for the manuscript.  Every function takes plain arrays / DataFrames (no torch needed) and saves a PNG.

PeerJ asks for figures as separate files named ``Figure N.png``; :func:`figure_path` follows that convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["figure_path", "plot_pareto", "plot_convergence", "plot_training_curves", "plot_confusion_matrix",
           "plot_roc", "plot_kl_distribution", "plot_gradcam_grid", "plot_ablation"]

DPI = 300
PALETTE = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3"]


def figure_path(out_dir: str | Path, number: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"Figure {number}.png"


def _save(fig, path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_pareto(archive: pd.DataFrame, front: pd.DataFrame, path, selected: Optional[Dict[str, float]] = None) -> Path:
    """Pairwise projections of the three-objective Pareto front (f1 vs f2, f1 vs f3, f2 vs f3)."""
    pairs = [("f2", "f1"), ("f3", "f1"), ("f2", "f3")]
    labels = {"f1": "Validation cross-entropy $f_1$", "f2": "GFLOPs $f_2$", "f3": "KL divergence $f_3$ (nats)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ok = archive[~archive.get("failed", pd.Series(False, index=archive.index)).astype(bool)]
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(ok[x], ok[y], s=12, c="#bbbbbb", label="evaluated")
        ax.scatter(front[x], front[y], s=28, c=PALETTE[0], label="Pareto front")
        if selected:
            ax.scatter([selected[x]], [selected[y]], s=120, marker="*", c=PALETTE[1], label="selected", zorder=5)
        ax.set_xlabel(labels[x]); ax.set_ylabel(labels[y]); ax.grid(alpha=.3)
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_convergence(conv: pd.DataFrame, path) -> Path:
    """Hypervolume (left) and best value of every objective (right) per generation."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    if "hypervolume" in conv and conv["hypervolume"].notna().any():
        axes[0].plot(conv["generation"], conv["hypervolume"], marker="o", ms=3, color=PALETTE[0])
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Hypervolume"); axes[0].grid(alpha=.3)
    for i, col in enumerate([c for c in conv.columns if c.startswith("best_")]):
        axes[1].plot(conv["generation"], conv[col] / max(conv[col].abs().max(), 1e-12), label=col[5:], color=PALETTE[i % 4])
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Best value (normalised)"); axes[1].legend(frameon=False); axes[1].grid(alpha=.3)
    return _save(fig, path)


def plot_training_curves(history: pd.DataFrame, path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=PALETTE[0])
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], label="validation", color=PALETTE[1])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(frameon=False); axes[0].grid(alpha=.3)
    axes[1].plot(history["epoch"], 100 * history["train_acc"], label="train", color=PALETTE[0])
    if "val_acc" in history:
        axes[1].plot(history["epoch"], 100 * history["val_acc"], label="validation", color=PALETTE[1])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(frameon=False); axes[1].grid(alpha=.3)
    return _save(fig, path)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: Sequence[str], path) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    return _save(fig, path)


def plot_roc(curves: Dict[str, Dict[str, list]], path, aucs: Optional[Dict[str, float]] = None) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, (name, c) in enumerate(curves.items()):
        lab = name if not aucs else f"{name} (AUC {aucs[name]:.3f})"
        ax.plot(c["fpr"], c["tpr"], label=lab, color=PALETTE[i % 4])
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=.3)
    return _save(fig, path)


def plot_kl_distribution(kl: np.ndarray, labels: np.ndarray, class_names: Sequence[str], path) -> Path:
    """Per-class distribution of KL(P_NAS || P_LR) on the test split."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.boxplot([kl[labels == c] for c in range(len(class_names))], showfliers=False)
    ax.set_xticks(range(1, len(class_names) + 1), list(class_names))    # version-independent tick labels
    ax.set_ylabel("KL divergence (nats)"); ax.grid(alpha=.3, axis="y")
    return _save(fig, path)


def plot_gradcam_grid(images: np.ndarray, cams: np.ndarray, titles: Sequence[str], path) -> Path:
    """Row of (image, overlay) pairs.  ``images`` (n, H, W) in [0, 1]; ``cams`` (n, H, W) in [0, 1]."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray"); axes[0, i].set_title(titles[i], fontsize=8)
        axes[1, i].imshow(images[i], cmap="gray"); axes[1, i].imshow(cams[i], cmap="jet", alpha=.45)
        for r in range(2):
            axes[r, i].axis("off")
    return _save(fig, path)


def plot_ablation(df: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(df["variant"], df["auc"], color=PALETTE[0]); ax.invert_yaxis()
    lo = max(0.0, float(df["auc"].min()) - 0.02)
    ax.set_xlim(lo, min(1.0, float(df["auc"].max()) + 0.01)); ax.set_xlabel("Macro AUC"); ax.grid(alpha=.3, axis="x")
    return _save(fig, path)
