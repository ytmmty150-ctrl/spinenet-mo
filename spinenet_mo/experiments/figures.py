"""Assemble every manuscript figure from the files written by the other experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import roc_curves
from ..utils import ensure_dir, get_logger, load_json
from ..viz import plots

LOG = get_logger()


def make_figures(run_dir: str | Path, class_names, out_dir: str | Path | None = None) -> list[Path]:
    """Write ``Figure N.png`` files for whichever result files exist in ``run_dir``."""
    run = Path(run_dir)
    out = ensure_dir(out_dir or run / "figures")
    made = []
    s = run / "search"
    if (s / "archive.csv").exists() and (s / "pareto_front.csv").exists():
        sel = load_json(s / "selected.json")["objectives"] if (s / "selected.json").exists() else None
        made.append(plots.plot_pareto(pd.read_csv(s / "archive.csv"), pd.read_csv(s / "pareto_front.csv"), plots.figure_path(out, 1), sel))
    if (s / "convergence.csv").exists():
        made.append(plots.plot_convergence(pd.read_csv(s / "convergence.csv"), plots.figure_path(out, 2)))
    f = run / "final"
    if (f / "history.csv").exists():
        made.append(plots.plot_training_curves(pd.read_csv(f / "history.csv"), plots.figure_path(out, 3)))
    if (f / "metrics.json").exists() and (f / "predictions.npz").exists():
        m, p = load_json(f / "metrics.json"), np.load(f / "predictions.npz")
        made.append(plots.plot_confusion_matrix(m["confusion_matrix"], class_names, plots.figure_path(out, 4)))
        aucs = {r["class"]: r["auc"] for r in m["per_class"] if r["class"] in class_names}
        made.append(plots.plot_roc(roc_curves(p["labels"], p["probs"], class_names), plots.figure_path(out, 5), aucs or None))
        if "kl" in p.files:
            made.append(plots.plot_kl_distribution(p["kl"], p["labels"], class_names, plots.figure_path(out, 6)))
    a = run / "ablation" / "ablation.csv"
    if a.exists():
        made.append(plots.plot_ablation(pd.read_csv(a), plots.figure_path(out, 7)))
    LOG.info("wrote %d figures to %s", len(made), out)
    return made
