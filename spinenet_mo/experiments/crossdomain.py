"""Cross-domain generalisation: fine-tune the DEXA-pretrained network on knee radiographs (Table 7)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ..data.datasets import make_loaders
from ..data.splits import build_index, summarize_splits
from ..evaluation.metrics import summarize
from ..training.trainer import fit, get_device, predict
from ..utils import ensure_dir, get_logger, save_json, set_seed
from .context import Context
from .final import load_trained, resolve_class_weights

LOG = get_logger()


def knee_splits(cfg, seed: int) -> pd.DataFrame:
    """70 / 15 / 15 stratified split of the knee dataset."""
    cd = cfg["cross_domain"]
    df = build_index(cd["data_root"], cfg["data"]["class_names"])
    idx = np.arange(len(df))
    keep, test = train_test_split(idx, test_size=cd["splits"]["test_frac"], stratify=df["label"], random_state=seed)
    val_rel = cd["splits"]["val_frac"] / (1 - cd["splits"]["test_frac"])
    train, val = train_test_split(keep, test_size=val_rel, stratify=df["label"].iloc[keep], random_state=seed)
    df["split"] = ""
    for name, ids in (("train", train), ("val", val), ("test", test)):
        df.loc[df.index[ids], "split"] = name
    return df


def run_cross_domain(ctx: Context, checkpoint: str | Path) -> dict:
    """Phase 1: train only the classification head; phase 2: fine-tune all layers (cosine annealing)."""
    cfg = ctx.cfg
    cd = cfg["cross_domain"]
    out = ensure_dir(Path(ctx.out) / "crossdomain")
    seed = cfg["seed"]
    set_seed(seed)
    device = get_device()
    df = knee_splits(cfg, seed)
    df.to_csv(out / "knee_splits.csv", index=False)
    LOG.info("knee split:\n%s", summarize_splits(df))
    loaders = make_loaders(df, cfg, reference_fn=None, seed=seed, batch_size=cfg["train"]["batch_size"])
    model, _ = load_trained(checkpoint, device)
    cw = resolve_class_weights(cfg["train"].get("class_weights", "auto"), df, len(ctx.class_names))
    wd = cd.get("weight_decay", 5e-4)

    for p in model.parameters():                                   # ---- phase 1: head only
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    h1 = fit(model, loaders, device, epochs=cd["phase1"]["epochs"], optimizer_name="adamw", lr_max=cd["phase1"]["lr"],
             lr_min=cd["phase1"]["lr"] * 0.01, weight_decay=wd, class_weights=cw, kl_weight=0.0)
    for p in model.parameters():                                   # ---- phase 2: all layers
        p.requires_grad = True
    h2 = fit(model, loaders, device, epochs=cd["phase2"]["epochs"], optimizer_name="adamw", lr_max=cd["phase2"]["lr"],
             lr_min=cd["phase2"]["lr"] * 0.01, weight_decay=wd, class_weights=cw, kl_weight=0.0)
    pd.DataFrame([{**r, "phase": 1} for r in h1] + [{**r, "phase": 2} for r in h2]).to_csv(out / "history.csv", index=False)

    test = predict(model, loaders["test"], device)
    metrics = summarize(test["labels"], test["probs"], ctx.class_names, cfg["evaluation"]["ci_level"])
    pd.DataFrame(metrics["per_class"]).to_csv(out / "per_class.csv", index=False)
    np.savez(out / "predictions.npz", probs=test["probs"], labels=test["labels"])
    save_json(metrics, out / "metrics.json")
    LOG.info("knee test accuracy %.4f | macro AUC %.4f (paper: 0.887 / 0.934)", metrics["accuracy"], metrics["macro_auc"])
    return metrics
