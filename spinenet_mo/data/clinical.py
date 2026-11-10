"""Clinical reference model ``P_LR`` and the KL objective ``f_3`` (Eq. 4).

``P_LR`` is a multinomial logistic regression fitted on the training split using four
standardised clinical risk factors (BMD T-score, age, body weight, prior fragility
fracture), L2-regularised with ``C = 1`` (scikit-learn).  It is kept fixed during the search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-8


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Row-wise ``KL(p || q) = sum_c p_c log(p_c / q_c)`` in nats."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


@dataclass
class ClinicalReference:
    """Fixed reference risk distribution, one probability vector per image."""

    scaler: Optional[StandardScaler]
    model: Optional[LogisticRegression]
    features: Sequence[str]
    prior: np.ndarray            # fallback when clinical features are unavailable
    val_accuracy: float = float("nan")

    @classmethod
    def fit(
        cls,
        clinical: Optional[pd.DataFrame],
        splits: pd.DataFrame,
        features: Sequence[str] = ("bmd_tscore", "age", "weight", "fracture_history"),
        C: float = 1.0,
        n_classes: int = 3,
    ) -> "ClinicalReference":
        """Fit on the training rows; report accuracy on the validation rows.

        ``clinical`` must contain ``image`` plus the feature columns.  If it is ``None`` the
        reference degrades to the training-set class prior (a *non-clinical* fallback that keeps the
        pipeline runnable but does not reproduce the clinical-alignment objective).
        """
        train_labels = splits.loc[splits["split"] == "train", "label"].to_numpy()
        prior = np.bincount(train_labels, minlength=n_classes).astype(float)
        prior /= prior.sum()
        if clinical is None:
            return cls(None, None, tuple(features), prior)
        merged = splits.merge(clinical, on="image", how="inner")
        tr, va = merged[merged["split"] == "train"], merged[merged["split"] == "val"]
        scaler = StandardScaler().fit(tr[list(features)].to_numpy(float))
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(scaler.transform(tr[list(features)].to_numpy(float)), tr["label"].to_numpy())
        acc = float("nan")
        if len(va):
            acc = float((model.predict(scaler.transform(va[list(features)].to_numpy(float))) == va["label"].to_numpy()).mean())
        return cls(scaler, model, tuple(features), prior, acc)

    @property
    def is_clinical(self) -> bool:
        return self.model is not None

    def predict_proba(self, clinical: Optional[pd.DataFrame], images: Sequence[str]) -> np.ndarray:
        """Reference distributions for ``images`` (rows in the same order)."""
        n_classes = len(self.prior)
        if not self.is_clinical or clinical is None:
            return np.tile(self.prior, (len(images), 1))
        table = clinical.set_index("image")
        missing = [im for im in images if im not in table.index]
        if missing:
            raise KeyError(f"{len(missing)} images have no clinical features (e.g. {missing[:3]})")
        X = table.loc[list(images), list(self.features)].to_numpy(float)
        probs = self.model.predict_proba(self.scaler.transform(X))
        full = np.zeros((len(images), n_classes))
        full[:, self.model.classes_] = probs
        return full

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ClinicalReference":
        import joblib

        return joblib.load(path)
