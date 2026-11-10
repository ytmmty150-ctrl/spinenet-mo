"""Final fine-tuning, evaluation and prediction export for one architecture."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..data.splits import class_weights
from ..evaluation.metrics import summarize
from ..models.spinenet import SpineNet
from ..search.arch import build_spec
from ..training.trainer import fit, get_device, predict
from ..utils import ensure_dir, get_logger, save_json, set_seed
from .context import Context

LOG = get_logger()


def resolve_class_weights(cfg_value, splits: pd.DataFrame, n_classes: int):
    if cfg_value in (None, "none"):
        return None
    if cfg_value == "auto":
        return class_weights(splits.loc[splits["split"] == "train", "label"].to_numpy(), n_classes)
    return list(cfg_value)


def train_final(ctx: Context, genome, num_blocks: int, out_dir, seed: Optional[int] = None,
                kl_weight: Optional[float] = None, spec_override: Optional[Dict] = None) -> Dict:
    """Fine-tune ``genome`` on the training split and evaluate on the held-out test split.

    Writes ``model.pt``, ``history.csv``, ``predictions.npz`` and ``metrics.json`` into ``out_dir`` and returns the
    metrics dictionary extended with the raw test probabilities / labels.
    """
    cfg = ctx.cfg
    t = cfg["train"]
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    out = ensure_dir(out_dir)
    device = get_device()
    spec = spec_override or build_spec(genome, num_blocks)
    model = SpineNet(spec)
    loaders = ctx.loaders(t["batch_size"], seed)
    cw = resolve_class_weights(t.get("class_weights", "auto"), ctx.splits, len(ctx.class_names))
    history = fit(
        model, loaders, device, epochs=t["epochs"], optimizer_name=t["optimizer"], lr_max=t["lr_max"], lr_min=t["lr_min"],
        weight_decay=t["weight_decay"], class_weights=cw,
        kl_weight=t.get("kl_weight", 0.0) if kl_weight is None else kl_weight, select_best_on=t.get("select_best_on", "val_acc"),
    )
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "genome": np.asarray(genome).tolist(), "num_blocks": num_blocks, "seed": seed},
               out / "model.pt")
    test = predict(model, loaders["test"], device)
    metrics = summarize(test["labels"], test["probs"], ctx.class_names, cfg["evaluation"]["ci_level"])
    metrics["seed"] = seed
    np.savez(out / "predictions.npz", probs=test["probs"], labels=test["labels"], kl=test["kl"], index=test["index"])
    save_json(metrics, out / "metrics.json")
    LOG.info("test accuracy %.4f | macro AUC %.4f", metrics["accuracy"], metrics["macro_auc"])
    return {**metrics, "probs": test["probs"], "labels": test["labels"], "model": model}


def load_trained(path, device=None):
    """Load a model saved by :func:`train_final`."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model = SpineNet(build_spec(np.asarray(ckpt["genome"]), ckpt["num_blocks"]))
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
