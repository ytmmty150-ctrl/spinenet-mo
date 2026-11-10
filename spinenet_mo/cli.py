"""Command-line interface:  ``spinenet-mo <command> [--config FILE] [--set key=value ...]``.

Typical order::

    prepare -> search -> train -> evaluate -> (ablation | seeds | crossdomain | baselines) -> efficiency -> figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .utils import get_logger, load_json, save_json

LOG = get_logger()
COMMANDS = ["prepare", "search", "train", "evaluate", "ablation", "seeds", "crossdomain", "baselines",
            "efficiency", "gradcam", "figures"]


def _selected(cfg):
    p = Path(cfg["output_dir"]) / "search" / "selected.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `spinenet-mo search` first")
    d = load_json(p)
    return d["genome"], d["num_blocks"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="spinenet-mo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides, e.g. search.generations=2 train.epochs=3")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cmd = args.command
    out = Path(cfg["output_dir"])

    if cmd == "prepare":
        from .experiments.context import prepare_data
        prepare_data(cfg)
        return 0

    from .experiments.context import load_context
    ctx = load_context(cfg)

    if cmd == "search":
        from .experiments.search import run_search
        run_search(ctx)
    elif cmd == "train":
        from .experiments.final import train_final
        genome, nb = _selected(cfg)
        train_final(ctx, genome, nb, out / "final")
    elif cmd == "evaluate":
        from .experiments.final import load_trained
        from .evaluation.metrics import summarize
        from .training.trainer import get_device, predict
        import numpy as np
        device = get_device()
        model, _ = load_trained(out / "final" / "model.pt", device)
        model.to(device)
        res = predict(model, ctx.loaders(cfg["train"]["batch_size"], cfg["seed"])["test"], device)
        m = summarize(res["labels"], res["probs"], ctx.class_names, cfg["evaluation"]["ci_level"])
        save_json(m, out / "final" / "metrics.json")
        np.savez(out / "final" / "predictions.npz", probs=res["probs"], labels=res["labels"], kl=res["kl"], index=res["index"])
        LOG.info("accuracy %.4f  macro AUC %.4f", m["accuracy"], m["macro_auc"])
    elif cmd == "ablation":
        from .experiments.ablation import run_ablation
        genome, nb = _selected(cfg)
        run_ablation(ctx, genome, nb)
    elif cmd == "seeds":
        from .experiments.seeds import run_seed_sensitivity
        genome, nb = _selected(cfg)
        run_seed_sensitivity(ctx, genome, nb)
    elif cmd == "crossdomain":
        from .experiments.crossdomain import run_cross_domain
        run_cross_domain(ctx, out / "final" / "model.pt")
    elif cmd == "baselines":
        from .experiments.baselines import run_baselines
        run_baselines(ctx, out / "final" / "predictions.npz")
    elif cmd == "efficiency":
        from .experiments.final import load_trained
        from .evaluation.efficiency import complexity_report
        from .search.arch import build_spec
        import numpy as np
        model, ck = load_trained(out / "final" / "model.pt")
        rep = complexity_report(model, build_spec(np.asarray(ck["genome"]), ck["num_blocks"]),
                                runs=cfg["evaluation"]["latency_runs"], warmup=cfg["evaluation"]["latency_warmup"])
        save_json(rep, out / "final" / "efficiency.json")
        LOG.info("%s", rep)
    elif cmd == "gradcam":
        from .experiments.gradcam_figure import make_gradcam_figure
        make_gradcam_figure(ctx, out / "final" / "model.pt", out / "figures")
    elif cmd == "figures":
        from .experiments.figures import make_figures
        make_figures(out, ctx.class_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
