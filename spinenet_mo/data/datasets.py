"""PyTorch datasets and transforms (images, optional ROI crop, clinical references)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def per_image_normalize(x):
    """Per-image intensity normalisation (zero mean, unit variance)."""
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + 1e-6)


def build_transforms(image_size: int = 224, train: bool = False, augment: Optional[Dict] = None, channels: int = 1):
    """Resize -> (augmentation) -> tensor -> per-image normalisation.

    Augmentation (training only): random horizontal flip, rotation within +-15 deg and uniform scaling in [0.85, 1.15].
    """
    import torchvision.transforms as T

    ops = [T.Grayscale(num_output_channels=channels), T.Resize((image_size, image_size))]
    if train:
        augment = augment or {}
        if augment.get("hflip", True):
            ops.append(T.RandomHorizontalFlip())
        ops.append(
            T.RandomAffine(degrees=augment.get("rotation_deg", 15), scale=tuple(augment.get("scale", (0.85, 1.15))))
        )
    ops += [T.ToTensor(), T.Lambda(per_image_normalize)]
    return T.Compose(ops)


def make_dataset_class():
    """Create the Dataset class lazily so that torch is only required when training."""
    from torch.utils.data import Dataset
    import torch

    class DexaDataset(Dataset):
        """Returns ``(image, label, reference_probs, index)``.

        ``reference`` holds P_LR for every image (used by the KL regulariser and by objective f_3);
        ``roi`` optionally maps an image key to a ``(x0, y0, x1, y1)`` box (L1-L4 region of interest).
        """

        def __init__(self, df: pd.DataFrame, transform, reference: Optional[np.ndarray] = None,
                     roi: Optional[pd.DataFrame] = None):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            n_classes = int(self.df["label"].max()) + 1
            self.reference = (
                np.full((len(self.df), n_classes), 1.0 / n_classes, dtype=np.float32)
                if reference is None else np.asarray(reference, dtype=np.float32)
            )
            self.roi = None if roi is None else roi.set_index("image")

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, i: int):
            row = self.df.iloc[i]
            img = Image.open(row["path"]).convert("L")
            if self.roi is not None and row["image"] in self.roi.index:
                b = self.roi.loc[row["image"]]
                img = img.crop((int(b.x0), int(b.y0), int(b.x1), int(b.y1)))
            x = self.transform(img)
            return x, int(row["label"]), torch.from_numpy(self.reference[i]), i

    return DexaDataset


def make_loaders(
    splits: pd.DataFrame,
    cfg,
    reference_fn=None,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    roi: Optional[pd.DataFrame] = None,
):
    """Create train / val / test loaders.

    ``reference_fn(image_keys) -> (n, n_classes)`` supplies the clinical reference distributions.
    Shuffling uses a seeded generator so that data order is reproducible.
    """
    import torch
    from torch.utils.data import DataLoader

    DexaDataset = make_dataset_class()
    data = cfg["data"]
    nw = data.get("num_workers", 0) if num_workers is None else num_workers
    loaders = {}
    for split in ("train", "val", "test"):
        sub = splits[splits["split"] == split]
        if sub.empty:
            continue
        ref = reference_fn(sub["image"].tolist()) if reference_fn is not None else None
        tf = build_transforms(data["image_size"], train=(split == "train"), augment=data.get("augment"),
                              channels=data.get("channels", 1))
        ds = DexaDataset(sub, tf, ref, roi)
        gen = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=nw,
            generator=gen if split == "train" else None, drop_last=False,
            worker_init_fn=lambda wid: np.random.seed(seed + wid), pin_memory=torch.cuda.is_available(),
        )
    return loaders
