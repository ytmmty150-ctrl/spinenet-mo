import numpy as np
import pandas as pd

from spinenet_mo.config import load_config
from spinenet_mo.viz import plots


def test_default_config_matches_paper_values():
    cfg = load_config("configs/default.yaml")
    assert cfg["seed"] == 42
    s = cfg["search"]
    assert (s["population_size"], s["mating_pool_size"], s["generations"], s["num_blocks"]) == (20, 10, 25, 4)
    assert s["crossover_prob"] == 0.6 and s["ref_divisions"] == 6 and s["candidate_epochs"] == 25
    assert cfg["train"]["epochs"] == 140 and cfg["train"]["batch_size"] == 32 and cfg["train"]["kl_weight"] == 0.1
    assert cfg["cross_domain"]["phase1"]["epochs"] == 30 and cfg["cross_domain"]["phase2"]["epochs"] == 60
    assert cfg.search.selection.max_gflops == 2.0


def test_overrides():
    cfg = load_config("configs/default.yaml", ["search.generations=2", "train.lr_max=0.01", "data.roi_csv=null"])
    assert cfg["search"]["generations"] == 2 and cfg["train"]["lr_max"] == 0.01 and cfg["data"]["roi_csv"] is None


def test_plots_write_pngs(tmp_path):
    rng = np.random.default_rng(0)
    archive = pd.DataFrame({"f1": rng.random(30), "f2": rng.random(30), "f3": rng.random(30), "failed": False})
    front = archive.iloc[:5]
    assert plots.plot_pareto(archive, front, plots.figure_path(tmp_path, 1), front.iloc[0].to_dict()).exists()
    conv = pd.DataFrame({"generation": range(5), "hypervolume": np.linspace(.1, .5, 5), "best_f1": np.linspace(1, .5, 5)})
    assert plots.plot_convergence(conv, plots.figure_path(tmp_path, 2)).name == "Figure 2.png"
    hist = pd.DataFrame({"epoch": range(1, 6), "train_loss": np.linspace(1, .2, 5), "val_loss": np.linspace(1, .3, 5),
                         "train_acc": np.linspace(.5, .95, 5), "val_acc": np.linspace(.5, .9, 5)})
    assert plots.plot_training_curves(hist, plots.figure_path(tmp_path, 3)).exists()
    names = ["Normal", "Osteopenia", "Osteoporosis"]
    assert plots.plot_confusion_matrix([[9, 1, 0], [1, 8, 1], [0, 2, 8]], names, plots.figure_path(tmp_path, 4)).exists()
    curves = {n: {"fpr": [0, .2, 1], "tpr": [0, .8, 1]} for n in names}
    assert plots.plot_roc(curves, plots.figure_path(tmp_path, 5), {n: .9 for n in names}).exists()
    assert plots.plot_kl_distribution(rng.random(60), np.repeat([0, 1, 2], 20), names, plots.figure_path(tmp_path, 6)).exists()
    assert plots.plot_gradcam_grid(rng.random((3, 16, 16)), rng.random((3, 16, 16)), names, plots.figure_path(tmp_path, 8)).exists()
    ab = pd.DataFrame({"variant": ["full", "no_eca"], "auc": [0.9, 0.88]})
    assert plots.plot_ablation(ab, plots.figure_path(tmp_path, 7)).exists()
