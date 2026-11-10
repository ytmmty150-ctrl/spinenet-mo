"""Ablation study (Table 6): remove one component at a time and test against the full model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..evaluation.stats import mcnemar_test, significance_letter
from ..search.analytic import cost_of_genome
from ..search.space import replace_conv_type, with_overrides
from ..utils import ensure_dir, get_logger, load_json, save_json
from .context import Context
from .final import train_final
from .search import run_search

LOG = get_logger()

VARIANTS = ["full", "no_ghostconv", "no_eca", "no_kl", "no_monas", "no_gap"]
LABELS = {
    "full": "Full SpineNet-MO",
    "no_ghostconv": "w/o GhostConv (standard conv)",
    "no_eca": "w/o ECA (no attention)",
    "no_kl": "w/o KL-divergence objective",
    "no_monas": "w/o MO-NAS (single-objective)",
    "no_gap": "w/o GAP (flatten only)",
}


def variant_genome(name: str, genome: np.ndarray, num_blocks: int) -> np.ndarray:
    """Genome of an ablated architecture (same weights-free structure, one component removed)."""
    if name in ("full", "no_kl"):
        return genome                                   # 'no_kl' only changes the training objective
    if name == "no_ghostconv":
        return replace_conv_type(genome, num_blocks, "ghost", "standard")
    if name == "no_eca":
        return with_overrides(genome, num_blocks, attention="none")
    if name == "no_gap":
        return with_overrides(genome, num_blocks, pooling="avg")      # adaptive pooling + flatten
    raise KeyError(name)


def run_ablation(ctx: Context, selected_genome, num_blocks: int, variants: List[str] | None = None) -> pd.DataFrame:
    cfg = ctx.cfg
    out = ensure_dir(Path(ctx.out) / "ablation")
    genome = np.asarray(selected_genome)
    variants = variants or VARIANTS
    results: Dict[str, Dict] = {}
    for name in variants:
        LOG.info("ablation variant: %s", name)
        if name == "no_monas":
            # single-objective NAS: search with f1 only and train its best architecture
            sel, _ = run_search(ctx, objectives=["f1"], tag="ablation/no_monas_search")
            g = np.asarray(sel["genome"])
        else:
            g = variant_genome(name, genome, num_blocks)
        kl = 0.0 if name == "no_kl" else None
        res = train_final(ctx, g, num_blocks, out / name, kl_weight=kl)
        cost = cost_of_genome(g, num_blocks, macs=cfg["search"].get("flops_counts_macs", True))
        results[name] = {"res": res, "cost": cost}

    ref = results["full"]["res"]
    rows = []
    for name in variants:
        r, cost = results[name]["res"], results[name]["cost"]
        row = {"variant": LABELS[name], "auc": r["macro_auc"], "accuracy_pct": 100 * r["accuracy"],
               "params_m": cost["params_m"], "gflops": cost["gflops"]}
        if name != "full":
            test = mcnemar_test(ref["labels"], ref["probs"].argmax(1), r["probs"].argmax(1))
            row.update({"p_value": test["p_value"], "mcnemar_chi2": test["chi2"], "sig": significance_letter(test["p_value"]),
                        "test_method": test["method"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    LOG.info("\n%s", df.to_string(index=False))
    return df
