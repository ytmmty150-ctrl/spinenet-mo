"""Image index and stratified train / validation / test splits."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_index(root: str | Path, class_names: Sequence[str]) -> pd.DataFrame:
    """Scan ``root/<class_name>/*`` and return a DataFrame ``[image, path, label, class_name]``.

    ``image`` is the path relative to ``root`` (used as the key of the optional clinical / ROI CSVs).
    """
    root = Path(root)
    rows = []
    for label, name in enumerate(class_names):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder {folder}")
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"image": str(p.relative_to(root)), "path": str(p), "label": label, "class_name": name})
    if not rows:
        raise RuntimeError(f"No images found under {root}")
    return pd.DataFrame(rows)


def _size(spec: float | int, n: int) -> int:
    return int(round(spec * n)) if isinstance(spec, float) and spec < 1 else int(spec)


def stratified_split(
    df: pd.DataFrame,
    test_size: float | int,
    val_size: float | int,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a ``split`` column (train / val / test) preserving the class distribution.

    ``test_size`` / ``val_size`` are absolute counts (ints) or fractions of the whole dataset (floats).
    The paper's 1,952-image dataset is split into 1,366 / 226 / 360 images with ``val_size=226``, ``test_size=360``.
    """
    n = len(df)
    n_test, n_val = _size(test_size, n), _size(val_size, n)
    idx = np.arange(n)
    trainval, test = train_test_split(idx, test_size=n_test, stratify=df["label"], random_state=seed)
    train, val = train_test_split(trainval, test_size=n_val, stratify=df["label"].iloc[trainval], random_state=seed)
    out = df.copy()
    out["split"] = ""
    out.iloc[train, out.columns.get_loc("split")] = "train"
    out.iloc[val, out.columns.get_loc("split")] = "val"
    out.iloc[test, out.columns.get_loc("split")] = "test"
    return out


def class_weights(labels: Sequence[int], n_classes: int = 3) -> np.ndarray:
    """Inverse-frequency ("balanced") weights ``N / (K * n_c)``.

    On the paper's training split (492 / 410 / 464 images) this gives 0.926 / 1.111 / 0.981, in line with the
    reported 0.927 / 1.111 / 0.980.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    return counts.sum() / (n_classes * np.maximum(counts, 1.0))


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
