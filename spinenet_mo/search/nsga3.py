"""NSGA-III (Deb & Jain, 2014) for integer genomes, implemented in NumPy.

The implementation follows Algorithm 1 of the manuscript:

1. sample ``N_pop`` genomes uniformly and evaluate them;
2. at every generation build a mating pool of size ``N_mate`` (binary tournaments on
   non-domination rank), create ``N_pop`` offspring by uniform crossover (``p_c``) and
   gene-wise mutation (``p_m = 1/d``), and evaluate them;
3. rank parents + offspring by fast non-dominated sorting, admit complete fronts until
   the last front ``F_l`` overflows the population budget, and fill the remaining slots
   from ``F_l`` through reference-point association and niching on the normalised
   objective space (Das-Dennis reference points on the unit simplex).

All objectives are **minimised**.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .space import crossover, genome_key, mutate

BIG_PENALTY = 1e3  # objective value assigned to candidates whose evaluation failed


# --------------------------------------------------------------------------- utilities
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[np.ndarray]:
    """Return the list of Pareto fronts (index arrays) of the objective matrix ``F``."""
    n = F.shape[0]
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i)
                dom_count[i] += 1
    fronts: List[np.ndarray] = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(np.asarray(current, dtype=np.int64))
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def das_dennis(n_obj: int, divisions: int) -> np.ndarray:
    """Structured reference points on the unit simplex (Das & Dennis, 1998)."""
    if n_obj == 1:
        return np.ones((1, 1))
    pts = []
    for comb in itertools.combinations(range(divisions + n_obj - 1), n_obj - 1):
        prev, row = -1, []
        for c in comb:
            row.append(c - prev - 1)
            prev = c
        row.append(divisions + n_obj - 2 - prev)
        pts.append(row)
    return np.asarray(pts, dtype=float) / divisions


def normalize_objectives(F: np.ndarray) -> np.ndarray:
    """NSGA-III normalisation: translate by the ideal point and scale by the hyper-plane intercepts."""
    m = F.shape[1]
    ideal = F.min(axis=0)
    Ft = F - ideal
    extremes = np.zeros((m, m))
    for i in range(m):
        w = np.full(m, 1e-6)
        w[i] = 1.0
        extremes[i] = Ft[np.argmin(np.max(Ft / w, axis=1))]
    intercepts = None
    try:
        plane = np.linalg.solve(extremes, np.ones(m))
        cand = 1.0 / plane
        if np.all(np.isfinite(cand)) and np.all(cand > 1e-6):
            intercepts = cand
    except np.linalg.LinAlgError:
        pass
    if intercepts is None:  # fall back to the worst values of the current set
        intercepts = Ft.max(axis=0)
    intercepts = np.where(intercepts < 1e-12, 1.0, intercepts)
    return Ft / intercepts


def associate(Fn: np.ndarray, refs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Associate every normalised point with the reference line of smallest perpendicular distance."""
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    unit = refs / np.where(norms == 0, 1, norms)
    proj = Fn @ unit.T                                       # (n, n_refs) scalar projections
    dist2 = np.maximum((Fn ** 2).sum(axis=1, keepdims=True) - proj ** 2, 0.0)
    dist = np.sqrt(dist2)
    niche = dist.argmin(axis=1)
    return niche, dist[np.arange(len(Fn)), niche]


def environmental_selection(F: np.ndarray, n_survive: int, refs: np.ndarray, rng: np.random.Generator):
    """Select ``n_survive`` individuals.  Returns ``(indices, ranks)`` for *all* rows of ``F``."""
    fronts = fast_non_dominated_sort(F)
    ranks = np.zeros(len(F), dtype=np.int64)
    for r, fr in enumerate(fronts):
        ranks[fr] = r
    chosen: List[int] = []
    last: Optional[np.ndarray] = None
    for fr in fronts:
        if len(chosen) + len(fr) <= n_survive:
            chosen.extend(fr.tolist())
        else:
            last = fr
            break
    if last is None or len(chosen) == n_survive:
        return np.asarray(chosen, dtype=np.int64), ranks

    pool = np.asarray(chosen + last.tolist(), dtype=np.int64)
    Fn = normalize_objectives(F[pool])
    niche, dist = associate(Fn, refs)
    n_chosen = len(chosen)
    rho = np.zeros(len(refs), dtype=np.int64)
    for j in niche[:n_chosen]:
        rho[j] += 1
    last_idx = list(range(n_chosen, len(pool)))                 # positions inside ``pool``
    available = {int(j) for j in niche[n_chosen:]}              # reference points with last-front members
    need = n_survive - n_chosen
    picked: List[int] = []
    while len(picked) < need:
        refs_avail = sorted(available)
        min_rho = min(rho[j] for j in refs_avail)
        ties = [j for j in refs_avail if rho[j] == min_rho]
        j = int(rng.choice(ties))
        members = [p for p in last_idx if niche[p] == j]
        if not members:
            available.discard(j)
            continue
        if rho[j] == 0:
            pick = min(members, key=lambda p: dist[p])
        else:
            pick = int(rng.choice(members))
        picked.append(pick)
        last_idx.remove(pick)
        rho[j] += 1
        if not any(niche[p] == j for p in last_idx):
            available.discard(j)
    survivors = chosen + [int(pool[p]) for p in picked]
    return np.asarray(survivors, dtype=np.int64), ranks


# --------------------------------------------------------------------------- hypervolume
def _hv2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = points[np.argsort(points[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact hyper-volume (minimisation) dominated by ``points`` w.r.t. ``ref`` for 1-3 objectives."""
    points = np.atleast_2d(points)
    keep = np.all(points < ref, axis=1)
    pts = points[keep]
    if len(pts) == 0:
        return 0.0
    m = pts.shape[1]
    if m == 1:
        return float(ref[0] - pts.min())
    if m == 2:
        return float(_hv2d(pts, ref))
    if m != 3:
        raise NotImplementedError("hypervolume implemented for at most 3 objectives")
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    hv = 0.0
    for k in range(len(pts)):
        z_next = pts[k + 1, 2] if k + 1 < len(pts) else ref[2]
        depth = z_next - pts[k, 2]
        if depth > 0:
            hv += depth * _hv2d(pts[: k + 1, :2], ref[:2])
    return float(hv)


# --------------------------------------------------------------------------- the search
@dataclass
class SearchResult:
    population: np.ndarray                  # genomes of the final population
    objectives: np.ndarray                  # their objective vectors
    pareto_idx: np.ndarray                  # indices of the first front inside ``population``
    history: List[Dict] = field(default_factory=list)
    archive: List[Dict] = field(default_factory=list)   # every unique evaluated architecture

    @property
    def pareto_genomes(self) -> np.ndarray:
        return self.population[self.pareto_idx]

    @property
    def pareto_objectives(self) -> np.ndarray:
        return self.objectives[self.pareto_idx]


EvalFn = Callable[[np.ndarray], Dict]


class NSGA3:
    """Evolutionary multi-objective search over integer genomes.

    Parameters
    ----------
    cardinalities : number of admissible values per gene.
    evaluate : callable ``genome -> {"objectives": [f1, f2, ...], ...aux}``; ``aux`` entries are
        kept in the archive (e.g. validation AUC used by the final selection rule).
    """

    def __init__(
        self,
        cardinalities: np.ndarray,
        evaluate: EvalFn,
        n_obj: int = 3,
        pop_size: int = 20,
        mating_pool: int = 10,
        generations: int = 25,
        crossover_prob: float = 0.6,
        mutation_prob: Optional[float] = None,
        ref_divisions: int = 6,
        seed: int = 42,
        logger=None,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.card = np.asarray(cardinalities, dtype=np.int64)
        self.evaluate_fn = evaluate
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.mating_pool = mating_pool
        self.generations = generations
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.refs = das_dennis(n_obj, ref_divisions)
        self.rng = np.random.default_rng(seed)
        self.logger = logger
        self.callback = callback
        self._cache: Dict[tuple, Dict] = {}

    # -- evaluation with caching -------------------------------------------------
    def _evaluate(self, genome: np.ndarray) -> np.ndarray:
        key = genome_key(genome)
        if key not in self._cache:
            try:
                out = dict(self.evaluate_fn(genome))
                obj = np.asarray(out["objectives"], dtype=float)
                if obj.shape != (self.n_obj,) or not np.all(np.isfinite(obj)):
                    raise ValueError(f"invalid objective vector {obj}")
            except Exception as exc:  # a failing candidate must not abort the whole search
                if self.logger:
                    self.logger.warning("candidate %s failed (%s); assigning penalty", key[:4], exc)
                out = {"objectives": [BIG_PENALTY] * self.n_obj, "failed": True}
                obj = np.asarray(out["objectives"], dtype=float)
            out["objectives"] = obj
            out["genome"] = np.asarray(genome)
            self._cache[key] = out
        return self._cache[key]["objectives"]

    def _eval_many(self, genomes: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self._evaluate(g) for g in genomes])

    # -- variation ------------------------------------------------------------------
    def _make_offspring(self, pop: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        n = len(pop)
        pool = []
        for _ in range(self.mating_pool):                      # binary tournaments on rank
            i, j = self.rng.integers(0, n, size=2)
            pool.append(i if ranks[i] <= ranks[j] else j)
        children = []
        while len(children) < self.pop_size:
            p1, p2 = self.rng.choice(pool, size=2, replace=len(pool) < 2)
            c1, c2 = crossover(pop[p1], pop[p2], self.rng, self.pc)
            children.append(mutate(c1, self.card, self.rng, self.pm))
            if len(children) < self.pop_size:
                children.append(mutate(c2, self.card, self.rng, self.pm))
        return np.stack(children)

    # -- main loop ------------------------------------------------------------------
    def run(self) -> SearchResult:
        pop = np.stack([np.asarray([self.rng.integers(0, c) for c in self.card]) for _ in range(self.pop_size)])
        F = self._eval_many(pop)
        ranks = np.zeros(len(pop), dtype=np.int64)
        for r, fr in enumerate(fast_non_dominated_sort(F)):
            ranks[fr] = r
        history = [self._snapshot(0, pop, F)]
        for g in range(1, self.generations + 1):
            off = self._make_offspring(pop, ranks)
            Fo = self._eval_many(off)
            allg, allF = np.vstack([pop, off]), np.vstack([F, Fo])
            idx, all_ranks = environmental_selection(allF, self.pop_size, self.refs, self.rng)
            pop, F = allg[idx], allF[idx]
            ranks = all_ranks[idx]
            snap = self._snapshot(g, pop, F)
            history.append(snap)
            if self.logger:
                self.logger.info(
                    "gen %02d | evaluated %4d | front size %2d | best objectives %s",
                    g, snap["n_evaluated"], snap["front_size"], np.round(snap["best"], 4).tolist(),
                )
            if self.callback:
                self.callback(snap)
        front = fast_non_dominated_sort(F)[0]
        archive = list(self._cache.values())
        return SearchResult(pop, F, front, history, archive)

    def _snapshot(self, gen: int, pop: np.ndarray, F: np.ndarray) -> Dict:
        front = fast_non_dominated_sort(F)[0]
        return {
            "generation": gen,
            "n_evaluated": len(self._cache),
            "front_size": int(len(front)),
            "best": F.min(axis=0),
            "objectives": F.copy(),
            "genomes": pop.copy(),
        }


def hypervolume_history(history: List[Dict], archive_objectives: np.ndarray, margin: float = 1.1) -> np.ndarray:
    """Hyper-volume of each generation's population on a common normalisation.

    Objectives are min-max normalised with the ideal / nadir of *all* evaluated architectures and
    the reference point is ``margin`` in every dimension, so values are comparable across generations.
    """
    lo, hi = archive_objectives.min(axis=0), archive_objectives.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    ref = np.full(archive_objectives.shape[1], margin)
    out = []
    for snap in history:
        Fn = (snap["objectives"] - lo) / span
        front = fast_non_dominated_sort(Fn)[0]
        out.append(hypervolume(Fn[front], ref))
    return np.asarray(out)
