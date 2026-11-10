"""Paired significance testing between classifiers (McNemar) and multi-run aggregation."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def mcnemar_test(y_true, pred_a, pred_b, continuity: bool = True, exact_below: int = 25) -> Dict:
    """McNemar's test on the paired correctness of two classifiers.

    ``b`` = samples only A classifies correctly, ``c`` = samples only B classifies correctly.
    With continuity correction ``chi2 = (|b - c| - 1)^2 / (b + c)`` (1 d.o.f.).  When the number of discordant
    pairs is below ``exact_below`` the exact two-sided binomial test is used instead.
    """
    y_true, a, b_ = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    a_ok, b_ok = a == y_true, b_ == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "method": "none (no discordant pairs)"}
    if n_disc < exact_below:
        p = float(stats.binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue)
        return {"b": b, "c": c, "chi2": float("nan"), "p_value": p, "method": "exact binomial"}
    diff = abs(b - c) - (1 if continuity else 0)
    chi2 = max(diff, 0) ** 2 / n_disc
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)),
            "method": "McNemar" + (" (continuity correction)" if continuity else "")}


def significance_letter(p: float) -> str:
    """Letter code used in the manuscript tables: a p<0.001, b p<0.01, c p<0.05, d p>=0.05 (n.s.)."""
    if p < 0.001:
        return "a"
    if p < 0.01:
        return "b"
    if p < 0.05:
        return "c"
    return "d"


def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def aggregate_runs(rows: Sequence[Dict[str, float]]) -> pd.DataFrame:
    """Mean and (sample) standard deviation of every numeric key across runs (seed sensitivity)."""
    df = pd.DataFrame(rows)
    return pd.DataFrame({"mean": df.mean(numeric_only=True), "std": df.std(numeric_only=True, ddof=1)})
