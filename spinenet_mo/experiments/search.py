"""Multi-objective architecture search (Algorithm 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..models.spinenet import SpineNet
from ..search.analytic import cost_of_genome
from ..search.arch import build_spec
from ..search.nsga3 import NSGA3, hypervolume_history
from ..search.objectives import CandidateEvaluator, select_final
from ..search.space import cardinalities, genome_to_dict
from ..training.trainer import get_device
from ..utils import ensure_dir, get_logger, save_json, set_seed
from .context import Context

LOG = get_logger()


def run_search(ctx: Context, objectives: Optional[Sequence[str]] = None, tag: str = "search", seed: Optional[int] = None):
    """Run NSGA-III and write the archive, Pareto front, convergence history and the selected architecture.

    Returns ``(selected_record, SearchResult)``.  With ``objectives=["f1"]`` the same machinery acts as a
    single-objective evolutionary search (used by the 'w/o multi-objective NAS' ablation).
    """
    cfg = ctx.cfg
    s = cfg["search"]
    objectives = list(objectives or s["objectives"])
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    out = ensure_dir(Path(ctx.out) / tag)
    device = get_device()
    L = s["num_blocks"]
    loaders = ctx.loaders(s["candidate_batch_size"], seed)
    evaluator = CandidateEvaluator(loaders, cfg, device, L, objectives, s.get("max_candidate_gflops"))
    search = NSGA3(
        cardinalities(L), evaluator, n_obj=len(objectives), pop_size=s["population_size"],
        mating_pool=s["mating_pool_size"], generations=s["generations"], crossover_prob=s["crossover_prob"],
        mutation_prob=s.get("mutation_prob"), ref_divisions=s["ref_divisions"], seed=seed, logger=LOG,
    )
    result = search.run()

    # ---- archive of every unique evaluated architecture -------------------------------------------------
    rows = []
    for rec in result.archive:
        row = {"genome": " ".join(map(str, rec["genome"])), "failed": bool(rec.get("failed", False)),
               "params": rec.get("params"), "gflops": rec.get("gflops"),
               "val_acc": rec.get("val_acc"), "val_auc": rec.get("val_auc")}
        for name, val in zip(objectives, rec["objectives"]):
            row[name] = float(val)
        rows.append(row)
    archive_df = pd.DataFrame(rows)
    archive_df.to_csv(out / "archive.csv", index=False)

    # ---- Pareto front and convergence ----------------------------------------------------------------------
    front = [dict(zip(objectives, map(float, o))) | {"genome": " ".join(map(str, g))}
             for o, g in zip(result.pareto_objectives, result.pareto_genomes)]
    pd.DataFrame(front).to_csv(out / "pareto_front.csv", index=False)
    ok = np.array([not r.get("failed", False) for r in result.archive])
    arch_obj = np.stack([r["objectives"] for r in result.archive])[ok]
    hv = hypervolume_history(result.history, arch_obj) if len(objectives) <= 3 and len(arch_obj) else np.array([])
    conv = pd.DataFrame({"generation": [h["generation"] for h in result.history],
                         "n_evaluated": [h["n_evaluated"] for h in result.history],
                         "hypervolume": hv if len(hv) else np.nan})
    for k, name in enumerate(objectives):
        conv[f"best_{name}"] = [float(h["best"][k]) for h in result.history]
    conv.to_csv(out / "convergence.csv", index=False)

    # ---- final architecture -------------------------------------------------------------------------------------
    if len(objectives) == 1:       # single-objective search: simply the best candidate
        best = min((r for r in result.archive if not r.get("failed")), key=lambda r: r["objectives"][0])
        selected = best
    else:
        selected = select_final(result.archive, s["selection"]["max_gflops"], s["selection"]["max_kl"],
                                pareto_only=result.pareto_genomes, objectives=objectives)
    if selected is None:
        raise RuntimeError("search produced no valid architecture")
    genome = np.asarray(selected["genome"])
    cost = cost_of_genome(genome, L, macs=s.get("flops_counts_macs", True))
    save_json({"genome": genome.tolist(), "num_blocks": L, "description": genome_to_dict(genome, L), "cost": cost,
               "objectives": dict(zip(objectives, map(float, selected["objectives"]))),
               "val_auc": selected.get("val_auc"), "unique_architectures_evaluated": len(result.archive)},
              out / "selected.json")
    LOG.info("selected architecture: %.2fM params, %.2f GFLOPs, val AUC %s", cost["params_m"], cost["gflops"], selected.get("val_auc"))
    return selected, result
