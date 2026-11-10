"""Manually designed baselines (ResNet-50, VGG-16, and CNN+RF hybrids) trained on identical splits and augmentation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from ..evaluation.metrics import summarize
from ..evaluation.stats import mcnemar_test, significance_letter
from ..models.baselines import build_baseline
from ..training.trainer import fit, get_device, predict
from ..utils import ensure_dir, get_logger, set_seed
from .context import Context
from .final import resolve_class_weights

LOG = get_logger()


@torch.no_grad()
def _extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y, _, _ in loader:
        feats.append(model.features(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def run_baselines(ctx: Context, reference_predictions: str | Path | None = None) -> pd.DataFrame:
    """Train every baseline, evaluate on the test split and (optionally) test against SpineNet-MO with McNemar."""
    cfg = ctx.cfg
    b = cfg["baselines"]
    out = ensure_dir(Path(ctx.out) / "baselines")
    device = get_device()
    seed = cfg["seed"]
    ref = np.load(reference_predictions) if reference_predictions and Path(reference_predictions).exists() else None
    trained: Dict[str, tuple] = {}
    rows = []
    for name in b["models"]:
        base = name.replace("_rf", "")
        set_seed(seed)
        model, size = build_baseline(name, len(ctx.class_names), b["pretrained"])
        local_cfg = copy.deepcopy(cfg)
        local_cfg["data"]["image_size"] = size
        if base not in trained:
            from ..data.datasets import make_loaders

            loaders = make_loaders(ctx.splits, local_cfg, ctx.reference_fn, seed=seed, batch_size=cfg["train"]["batch_size"],
                                   roi=ctx.roi)
            cw = resolve_class_weights(cfg["train"].get("class_weights", "auto"), ctx.splits, len(ctx.class_names))
            history = fit(model, loaders, device, epochs=b["epochs"], optimizer_name="adamw", lr_max=b["lr"], lr_min=b["lr"] * 0.01,
                          weight_decay=cfg["train"]["weight_decay"], class_weights=cw)
            pd.DataFrame(history).to_csv(out / f"{base}_history.csv", index=False)
            trained[base] = (model, loaders)
        model, loaders = trained[base]
        if name.endswith("_rf"):
            Xtr, ytr = _extract(model, loaders["train"], device)
            Xte, yte = _extract(model, loaders["test"], device)
            rf = RandomForestClassifier(n_estimators=b["rf_trees"], random_state=seed, n_jobs=-1).fit(Xtr, ytr)
            probs, labels = rf.predict_proba(Xte), yte
        else:
            res = predict(model, loaders["test"], device)
            probs, labels = res["probs"], res["labels"]
        m = summarize(labels, probs, ctx.class_names, cfg["evaluation"]["ci_level"])
        row = {"model": name, "accuracy_pct": 100 * m["accuracy"], "macro_sensitivity_pct": 100 * m["macro_sensitivity"],
               "macro_specificity_pct": 100 * m["macro_specificity"], "macro_f1_pct": 100 * m["macro_f1"], "auc": m["macro_auc"]}
        if ref is not None:
            t = mcnemar_test(ref["labels"], ref["probs"].argmax(1), probs.argmax(1))
            row.update({"p_vs_spinenet_mo": t["p_value"], "sig": significance_letter(t["p_value"]), "test_method": t["method"]})
        rows.append(row)
        np.savez(out / f"{name}_predictions.npz", probs=probs, labels=labels)
    df = pd.DataFrame(rows)
    df.to_csv(out / "baselines.csv", index=False)
    LOG.info("\n%s", df.to_string(index=False))
    return df
