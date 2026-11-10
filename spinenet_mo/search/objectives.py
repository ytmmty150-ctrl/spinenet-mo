"""Objective evaluation of one candidate architecture (f1, f2, f3) -- Section 3.2."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..utils import get_logger
from .analytic import cost_of_genome
from .arch import build_spec

LOG = get_logger()


class CandidateEvaluator:
    """Callable ``genome -> {"objectives": [...], ...}`` used by :class:`~spinenet_mo.search.nsga3.NSGA3`.

    * ``f1`` -- validation categorical cross-entropy after short training (Eq. 2);
    * ``f2`` -- analytic FLOPs in giga-operations for a 224 x 224 x 1 input (Eq. 3);
    * ``f3`` -- mean KL divergence between the model's predicted distribution and the clinical reference
      ``P_LR`` on the validation split (Eq. 4).

    ``objectives`` selects (and orders) the objectives, e.g. ``["f1"]`` for the single-objective NAS ablation.
    Training is skipped when neither ``f1`` nor ``f3`` is requested, or when the candidate exceeds
    ``max_gflops`` (it is then assigned a penalty; this keeps the search tractable).
    """

    def __init__(self, loaders, cfg, device, num_blocks: int, objectives: Sequence[str] = ("f1", "f2", "f3"),
                 max_gflops: float | None = None):
        self.loaders, self.cfg, self.device = loaders, cfg, device
        self.num_blocks = num_blocks
        self.objectives = list(objectives)
        self.max_gflops = max_gflops
        self.n_calls = 0

    def __call__(self, genome: np.ndarray) -> Dict:
        s = self.cfg["search"]
        cost = cost_of_genome(genome, self.num_blocks, macs=s.get("flops_counts_macs", True))
        gflops = cost["gflops"]
        out: Dict = {"gflops": gflops, "params": cost["params"]}
        need_training = any(o in ("f1", "f3") for o in self.objectives)
        values = {"f2": gflops}
        if need_training:
            if self.max_gflops is not None and gflops > self.max_gflops:
                raise ValueError(f"{gflops:.2f} GFLOPs exceeds the candidate cap of {self.max_gflops}")
            f1, f3, acc, auc = self._train_and_score(genome)
            values.update({"f1": f1, "f3": f3})
            out.update({"val_acc": acc, "val_auc": auc})
        out["objectives"] = [values[o] for o in self.objectives]
        out["all_objectives"] = values
        self.n_calls += 1
        return out

    def _train_and_score(self, genome: np.ndarray):
        import torch
        from sklearn.metrics import roc_auc_score

        from ..models.spinenet import SpineNet
        from ..training.trainer import fit, predict

        s = self.cfg["search"]
        spec = build_spec(genome, self.num_blocks)
        model = SpineNet(spec)
        hp = spec["train_hparams"]
        history = fit(
            model, self.loaders, self.device, epochs=s["candidate_epochs"], optimizer_name=hp["optimizer"],
            lr_max=hp["lr"], lr_min=hp["lr"] * s.get("lr_min_ratio", 0.01), weight_decay=s.get("weight_decay", 5e-4),
            class_weights=None, kl_weight=0.0, verbose=False,
        )
        v = predict(model, self.loaders["val"], self.device)
        f1 = float(v["ce"].mean())
        f3 = float(v["kl"].mean())
        acc = float((v["probs"].argmax(1) == v["labels"]).mean())
        try:
            auc = float(roc_auc_score(v["labels"], v["probs"], multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f1, f3, acc, auc


def select_final(archive: List[Dict], max_gflops: float = 2.0, max_kl: float = 0.05,
                 pareto_only: np.ndarray | None = None, objectives: Sequence[str] = ("f1", "f2", "f3")) -> Dict | None:
    """Final-architecture rule of the paper: maximise AUC over the Pareto set subject to
    ``f2 <= max_gflops`` and ``f3 <= max_kl``.

    ``archive`` entries need ``all_objectives`` and ``val_auc``.  If no architecture satisfies the constraints the
    one with the smallest violation is returned (a warning is logged) instead of failing.
    """
    candidates = [a for a in archive if not a.get("failed") and "val_auc" in a]
    if pareto_only is not None:
        keys = {tuple(int(x) for x in g) for g in pareto_only}
        candidates = [a for a in candidates if tuple(int(x) for x in a["genome"]) in keys]
    if not candidates:
        return None
    ok = [a for a in candidates if a["all_objectives"]["f2"] <= max_gflops and a["all_objectives"].get("f3", 0.0) <= max_kl]
    if ok:
        return max(ok, key=lambda a: (a["val_auc"], -a["all_objectives"]["f2"]))
    LOG.warning("no Pareto architecture satisfies f2<=%.2f and f3<=%.3f; returning the least-violating one", max_gflops, max_kl)
    def violation(a):
        return max(0.0, a["all_objectives"]["f2"] - max_gflops) / max_gflops + max(0.0, a["all_objectives"].get("f3", 0.0) - max_kl) / max_kl
    return min(candidates, key=violation)
