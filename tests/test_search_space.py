import numpy as np

from spinenet_mo.search import space
from spinenet_mo.search.analytic import cost_of_genome
from spinenet_mo.search.arch import build_spec, eca_kernel_size

L = 4


def test_genome_shapes_and_bounds():
    rng = np.random.default_rng(0)
    card = space.cardinalities(L)
    assert len(card) == space.genome_length(L)
    for _ in range(50):
        g = space.random_genome(L, rng)
        assert g.shape == card.shape and np.all(g >= 0) and np.all(g < card)


def test_mutation_stays_in_bounds_and_changes_something():
    rng = np.random.default_rng(1)
    card = space.cardinalities(L)
    g = space.random_genome(L, rng)
    changed = 0
    for _ in range(100):
        m = space.mutate(g, card, rng, 1.0 / len(g))
        assert np.all(m >= 0) and np.all(m < card)
        changed += int(np.any(m != g))
    assert changed > 0


def test_crossover_children_are_mixtures():
    rng = np.random.default_rng(2)
    a, b = np.zeros(20, dtype=int), np.ones(20, dtype=int)
    c1, c2 = space.crossover(a, b, rng, 1.0)
    assert np.all(c1 + c2 == 1)


def test_overrides_and_replacement():
    rng = np.random.default_rng(3)
    g = space.random_genome(L, rng)
    d = space.genome_to_dict(space.with_overrides(g, L, attention="none"), L)
    assert d["attention"] == "none"
    d = space.genome_to_dict(space.with_overrides(g, L, pooling="avg"), L)
    assert d["pooling"] == "avg"
    g2 = space.replace_conv_type(g, L, "ghost", "standard")
    assert all(b["conv_type"] != "ghost" for b in space.genome_to_dict(g2, L)["blocks"])


def test_spec_and_cost_are_deterministic_and_positive():
    rng = np.random.default_rng(4)
    for _ in range(20):
        g = space.random_genome(L, rng)
        assert build_spec(g, L)["blocks"], "decoded architecture has blocks"
        c1, c2 = cost_of_genome(g, L), cost_of_genome(g, L)
        assert c1 == c2 and c1["params"] > 0 and c1["gflops"] > 0


def test_ghost_is_cheaper_than_standard():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(30):
        g = space.random_genome(L, rng)
        g_std = space.replace_conv_type(g, L, "ghost", "standard")
        if np.any(g != g_std):
            wins += cost_of_genome(g, L)["params"] <= cost_of_genome(g_std, L)["params"]
    assert wins > 0


def test_eca_kernel_is_odd():
    for c in (16, 32, 64, 128, 256, 512):
        k = eca_kernel_size(c)
        assert k % 2 == 1 and k >= 3 or k == 1


def test_gap_flag_changes_head():
    rng = np.random.default_rng(6)
    g = space.with_overrides(space.random_genome(L, rng), L, pooling="global_avg")
    ops = [l["op"] for l in build_spec(g, L)["head"]]
    assert ops[0] == "gap" and ops[1] == "flatten"
    ops = [l["op"] for l in build_spec(space.with_overrides(g, L, pooling="avg"), L)["head"]]
    assert ops[0] == "adaptive_pool"
