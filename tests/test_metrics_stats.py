import numpy as np
import pytest
from scipy import stats as sps

from spinenet_mo.evaluation import metrics, stats


def test_wilson_interval_known_value():
    lo, hi = metrics.wilson_ci(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.96
    assert metrics.wilson_ci(0, 10)[0] == 0.0


def test_hanley_mcneil_contains_auc():
    lo, hi = metrics.hanley_mcneil_ci(0.9, 100, 200)
    assert lo < 0.9 < hi <= 1.0


def test_summary_on_perfect_and_random_predictions():
    y = np.repeat([0, 1, 2], 30)
    perfect = np.eye(3)[y]
    m = metrics.summarize(y, perfect, ["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["macro_auc"] == 1.0 and m["macro_sensitivity"] == 1.0
    assert np.trace(np.array(m["confusion_matrix"])) == 90
    rng = np.random.default_rng(0)
    r = metrics.summarize(y, rng.dirichlet(np.ones(3), 90), ["a", "b", "c"])
    assert 0.3 < r["macro_auc"] < 0.7


def test_roc_curves_endpoints():
    y = np.array([0, 1, 2, 0, 1, 2])
    c = metrics.roc_curves(y, np.eye(3)[y], ["a", "b", "c"])
    assert set(c) == {"a", "b", "c"} and c["a"]["tpr"][-1] == 1.0


def test_mcnemar_matches_formula():
    y = np.zeros(200, dtype=int)
    a = y.copy(); b = y.copy()
    a[:30] = 1                       # A wrong on 30
    b[20:35] = 1                     # B wrong on 15 (10 overlap)
    res = stats.mcnemar_test(y, a, b)
    only_a_wrong, only_b_wrong = 20, 5
    assert (res["b"], res["c"]) == (only_b_wrong, only_a_wrong)           # b: only A correct
    chi2 = (abs(only_a_wrong - only_b_wrong) - 1) ** 2 / 25
    assert res["method"].startswith("McNemar") or res["method"] == "exact binomial"
    if res["method"].startswith("McNemar"):
        assert np.isclose(res["chi2"], chi2) and np.isclose(res["p_value"], sps.chi2.sf(chi2, 1))


def test_mcnemar_identical_classifiers():
    y = np.array([0, 1, 1, 0]); r = stats.mcnemar_test(y, y, y)
    assert r["p_value"] == 1.0


def test_exact_binomial_for_few_discordant():
    y = np.zeros(50, dtype=int); a = y.copy(); b = y.copy(); b[:5] = 1
    r = stats.mcnemar_test(y, a, b)
    assert r["method"] == "exact binomial" and np.isclose(r["p_value"], 2 * 0.5 ** 5)


@pytest.mark.parametrize("p,letter", [(0.0005, "a"), (0.005, "b"), (0.03, "c"), (0.5, "d")])
def test_letter_codes(p, letter):
    assert stats.significance_letter(p) == letter


def test_format_p_and_aggregate():
    assert stats.format_p(1e-5) == "p < 0.001" and stats.format_p(0.0123) == "p = 0.012"
    df = stats.aggregate_runs([{"auc": 0.9, "acc": 90}, {"auc": 0.8, "acc": 80}, {"auc": 1.0, "acc": 100}])
    assert np.isclose(df.loc["auc", "mean"], 0.9) and np.isclose(df.loc["auc", "std"], 0.1)
