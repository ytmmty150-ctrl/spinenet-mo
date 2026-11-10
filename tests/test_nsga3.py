import numpy as np

from spinenet_mo.search import nsga3


def test_dominance_and_sorting():
    assert nsga3.dominates(np.array([1, 1]), np.array([2, 2]))
    assert not nsga3.dominates(np.array([1, 2]), np.array([2, 1]))
    F = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]], dtype=float)
    fronts = nsga3.fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert fronts[-1].tolist() == [4]


def test_das_dennis_counts():
    assert nsga3.das_dennis(3, 6).shape == (28, 3)           # C(8, 2) reference points, as in the paper
    pts = nsga3.das_dennis(3, 6)
    assert np.allclose(pts.sum(axis=1), 1.0)
    assert nsga3.das_dennis(2, 4).shape == (5, 2)


def test_hypervolume_exact():
    ref = np.array([1.0, 1.0, 1.0])
    assert np.isclose(nsga3.hypervolume(np.array([[0.5, 0.5, 0.5]]), ref), 0.125)
    assert np.isclose(nsga3.hypervolume(np.array([[0.2, 0.2]]), np.array([1.0, 1.0])), 0.64)
    # dominated points do not change the volume
    a = nsga3.hypervolume(np.array([[0.2, 0.2, 0.2]]), ref)
    b = nsga3.hypervolume(np.array([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]]), ref)
    assert np.isclose(a, b)


def test_hypervolume_two_boxes_inclusion_exclusion():
    ref = np.array([1.0, 1.0, 1.0])
    pts = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    v1, v2 = 1.0 * 0.5 * 0.5, 0.5 * 1.0 * 0.5
    overlap = 0.5 * 0.5 * 0.5
    assert np.isclose(nsga3.hypervolume(pts, ref), v1 + v2 - overlap)


def test_environmental_selection_size_and_unique():
    rng = np.random.default_rng(0)
    F = rng.random((40, 3))
    idx, ranks = nsga3.environmental_selection(F, 20, nsga3.das_dennis(3, 6), rng)
    assert len(idx) == 20 and len(set(idx.tolist())) == 20 and len(ranks) == 40


def test_search_improves_on_toy_problem_and_caches():
    card = np.full(10, 8)
    calls = []

    def evaluate(g):
        calls.append(tuple(g))
        x = g / 7.0
        return {"objectives": [x[:5].sum(), (1 - x[5:]).sum(), np.abs(x[:5] - x[5:]).sum()]}

    s = nsga3.NSGA3(card, evaluate, n_obj=3, pop_size=20, mating_pool=10, generations=15, seed=0)
    res = s.run()
    assert len(calls) == len(set(calls)), "every genome is evaluated at most once (cache)"
    first, last = res.history[0]["objectives"], res.history[-1]["objectives"]
    assert last.sum(axis=1).min() <= first.sum(axis=1).min()
    assert len(res.pareto_genomes) >= 1
    hv = nsga3.hypervolume_history(res.history, np.stack([r["objectives"] for r in res.archive]))
    assert len(hv) == len(res.history) and hv[-1] >= hv[0] - 1e-9


def test_failed_candidates_get_penalty():
    def evaluate(g):
        if g[0] == 0:
            raise RuntimeError("boom")
        return {"objectives": [float(g.sum()), float(len(g) - g.sum())]}

    res = nsga3.NSGA3(np.full(4, 3), evaluate, n_obj=2, pop_size=8, mating_pool=4, generations=3, ref_divisions=4, seed=1).run()
    failed = [r for r in res.archive if r.get("failed")]
    assert failed and all(r["genome"][0] == 0 for r in failed)
    assert np.all(np.isfinite(res.pareto_objectives))


def test_search_is_reproducible():
    def evaluate(g):
        return {"objectives": [float(g.sum()), float((g ** 2).sum())]}

    runs = [nsga3.NSGA3(np.full(6, 5), evaluate, n_obj=2, pop_size=10, mating_pool=6, generations=4, ref_divisions=5, seed=7).run()
            for _ in range(2)]
    assert np.array_equal(runs[0].population, runs[1].population)
