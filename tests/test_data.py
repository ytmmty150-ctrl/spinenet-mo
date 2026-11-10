import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spinenet_mo.data import clinical, splits


@pytest.fixture()
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    for name, n in (("Normal", 40), ("Osteopenia", 30), ("Osteoporosis", 35)):
        (tmp_path / name).mkdir()
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).save(tmp_path / name / f"{i}.png")
    return tmp_path


def test_index_and_stratified_split(dataset):
    df = splits.build_index(dataset, ["Normal", "Osteopenia", "Osteoporosis"])
    assert len(df) == 105 and df["label"].nunique() == 3
    out = splits.stratified_split(df, 20, 15, seed=1)
    assert (out["split"] == "test").sum() == 20 and (out["split"] == "val").sum() == 15
    assert set(out["split"]) == {"train", "val", "test"}
    assert out.groupby("split")["label"].nunique().eq(3).all()
    again = splits.stratified_split(df, 20, 15, seed=1)
    assert out["split"].tolist() == again["split"].tolist()
    assert splits.summarize_splits(out).to_numpy().sum() == 105


def test_class_weights_match_paper_example():
    labels = np.repeat([0, 1, 2], [492, 410, 464])
    w = splits.class_weights(labels)
    assert np.allclose(w, [0.926, 1.111, 0.981], atol=2e-3)


def test_kl_divergence_properties():
    p = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
    assert np.allclose(clinical.kl_divergence(p, p), 0, atol=1e-9)
    q = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    assert np.all(clinical.kl_divergence(p, q) > 0)
    manual = (p * np.log(p / q)).sum(1)
    assert np.allclose(clinical.kl_divergence(p, q), manual, atol=1e-6)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    clin = pd.DataFrame({"image": [f"i{k}" for k in range(n)], "bmd_tscore": -1.5 * labels + rng.normal(0, 1, n),
                         "age": rng.normal(60, 8, n), "weight": rng.normal(70, 9, n), "fracture_history": rng.integers(0, 2, n)})
    sp = pd.DataFrame({"image": clin["image"], "label": labels, "class_name": labels.astype(str),
                       "split": np.where(np.arange(n) < 200, "train", np.where(np.arange(n) < 250, "val", "test"))})
    return clin, sp


def test_clinical_reference_fit_predict_save(tmp_path):
    clin, sp = _toy()
    ref = clinical.ClinicalReference.fit(clin, sp)
    assert ref.is_clinical and ref.val_accuracy > 0.5
    p = ref.predict_proba(clin, sp["image"].tolist()[:10])
    assert p.shape == (10, 3) and np.allclose(p.sum(1), 1)
    ref.save(str(tmp_path / "ref.joblib"))
    again = clinical.ClinicalReference.load(str(tmp_path / "ref.joblib"))
    assert np.allclose(again.predict_proba(clin, sp["image"].tolist()[:10]), p)


def test_fallback_prior_without_clinical_table():
    _, sp = _toy()
    ref = clinical.ClinicalReference.fit(None, sp)
    p = ref.predict_proba(None, ["a", "b"])
    assert not ref.is_clinical and np.allclose(p.sum(1), 1) and p.shape == (2, 3)
