"""Shared experiment context: splits, clinical reference, data loaders and output folders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.clinical import ClinicalReference
from ..data.splits import build_index, stratified_split, summarize_splits
from ..utils import ensure_dir, get_logger

LOG = get_logger()


@dataclass
class Context:
    cfg: dict
    splits: pd.DataFrame
    clinical: Optional[pd.DataFrame]
    reference: ClinicalReference
    roi: Optional[pd.DataFrame]
    out: Path

    @property
    def class_names(self):
        return list(self.cfg["data"]["class_names"])

    def reference_fn(self, images):
        return self.reference.predict_proba(self.clinical, images)

    def loaders(self, batch_size: int, seed: int, num_workers: Optional[int] = None):
        from ..data.datasets import make_loaders

        return make_loaders(self.splits, self.cfg, self.reference_fn, seed=seed, batch_size=batch_size,
                            num_workers=num_workers, roi=self.roi)


def prepare_data(cfg: dict) -> pd.DataFrame:
    """Index the image folders, create the stratified split and write ``splits_csv``."""
    d = cfg["data"]
    df = build_index(d["root"], d["class_names"])
    df = stratified_split(df, d["test_size"], d["val_size"], cfg["seed"])
    Path(d["splits_csv"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(d["splits_csv"], index=False)
    LOG.info("split summary:\n%s", summarize_splits(df))
    return df


def load_context(cfg: dict) -> Context:
    d = cfg["data"]
    if not Path(d["splits_csv"]).exists():
        raise FileNotFoundError(f"{d['splits_csv']} not found - run `spinenet-mo prepare` first")
    splits = pd.read_csv(d["splits_csv"])
    clinical = pd.read_csv(d["clinical_csv"]) if d.get("clinical_csv") else None
    roi = pd.read_csv(d["roi_csv"]) if d.get("roi_csv") else None
    cr = cfg["clinical_reference"]
    reference = ClinicalReference.fit(clinical, splits, cr["features"], cr["C"], len(d["class_names"]))
    if reference.is_clinical:
        LOG.info("clinical reference fitted: validation accuracy %.3f (paper: 0.743)", reference.val_accuracy)
    else:
        LOG.warning("no clinical_csv given: f3/KL uses the class-prior fallback instead of the clinical logistic regression")
    return Context(cfg, splits, clinical, reference, roi, ensure_dir(cfg["output_dir"]))
