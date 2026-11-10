"""Seed-sensitivity analysis (Table 3): re-train the selected architecture under N random seeds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.stats import aggregate_runs
from ..utils import ensure_dir, get_logger, save_json
from .context import Context
from .final import train_final

LOG = get_logger()


def run_seed_sensitivity(ctx: Context, genome, num_blocks: int) -> pd.DataFrame:
    cfg = ctx.cfg
    out = ensure_dir(Path(ctx.out) / "seeds")
    rows = []
    for i in range(cfg["evaluation"]["n_seeds"]):
        seed = cfg["seed"] + i
        res = train_final(ctx, genome, num_blocks, out / f"seed_{seed}", seed=seed)
        rows.append({"seed": seed, "accuracy_pct": 100 * res["accuracy"], "auc": res["macro_auc"],
                     "macro_sensitivity_pct": 100 * res["macro_sensitivity"],
                     "macro_specificity_pct": 100 * res["macro_specificity"]})
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    summary = aggregate_runs(rows).drop(index="seed", errors="ignore")
    summary.to_csv(out / "summary.csv")
    LOG.info("seed sensitivity (mean / std):\n%s", summary)
    return summary
