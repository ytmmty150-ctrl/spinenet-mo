"""Classification metrics (accuracy, per-class sensitivity / specificity / precision / F1, one-vs-rest AUC)
with confidence intervals (Wilson for proportions, Hanley-McNeil for AUC)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def wilson_ci(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int, level: float = 0.95) -> tuple[float, float]:
    """Hanley & McNeil (1982) normal-approximation interval for the AUC."""
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    z = stats.norm.ppf(0.5 + level / 2)
    se = np.sqrt(max(var, 0.0))
    return float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def per_class_table(y_true, y_pred, probs, class_names: Sequence[str], level: float = 0.95) -> pd.DataFrame:
    """One row per class (one-vs-rest) plus a macro-average row."""
    y_true, y_pred, probs = np.asarray(y_true), np.asarray(y_pred), np.asarray(probs)
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    rows = []
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else np.nan
        onevsrest = (y_true == c).astype(int)
        auc = roc_auc_score(onevsrest, probs[:, c]) if 0 < onevsrest.sum() < len(onevsrest) else np.nan
        lo, hi = hanley_mcneil_ci(auc, int(onevsrest.sum()), int(len(onevsrest) - onevsrest.sum()), level)
        sl, sh = wilson_ci(int(tp), int(tp + fn), level)
        rows.append({"class": name, "support": int(tp + fn), "sensitivity": sens, "sens_ci_low": sl, "sens_ci_high": sh,
                     "specificity": spec, "precision": prec, "f1": f1, "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi})
    df = pd.DataFrame(rows)
    macro = {"class": "macro", "support": int(df["support"].sum())}
    for col in ("sensitivity", "specificity", "precision", "f1", "auc"):
        macro[col] = float(df[col].mean())
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def summarize(y_true, probs, class_names: Sequence[str], level: float = 0.95) -> Dict:
    """All headline metrics for a set of predictions."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    n = len(y_true)
    correct = int((y_pred == y_true).sum())
    table = per_class_table(y_true, y_pred, probs, class_names, level)
    macro = table.iloc[-1]
    try:
        macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(len(class_names)))))
    except ValueError:
        macro_auc = float("nan")
    lo, hi = wilson_ci(correct, n, level)
    return {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci": [lo, hi],
        "macro_sensitivity": float(macro["sensitivity"]),
        "macro_specificity": float(macro["specificity"]),
        "macro_precision": float(macro["precision"]),
        "macro_f1": float(macro["f1"]),
        "macro_auc": macro_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(class_names)).tolist(),
        "per_class": table.to_dict(orient="records"),
    }


def roc_curves(y_true, probs, class_names: Sequence[str]) -> Dict[str, Dict[str, list]]:
    """One-vs-rest ROC curve points per class (for plotting)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    out = {}
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), probs[:, c])
        out[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return out
