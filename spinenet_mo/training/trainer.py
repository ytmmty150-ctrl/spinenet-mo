"""Training and prediction for any ``nn.Module`` producing 3-class logits."""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_logger

LOG = get_logger()


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


def kl_to_reference(logits: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample ``KL(P_NAS || P_LR)`` in nats (Eq. 4); ``P_NAS`` is the softmax of the logits."""
    p = F.softmax(logits, dim=1).clamp_min(eps)
    q = reference.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    """Softmax probabilities, labels, dataset indices and mean loss / KL on a loader."""
    model.eval()
    probs, labels, idx, kls, ces = [], [], [], [], []
    for x, y, ref, i in loader:
        x, y, ref = x.to(device), y.to(device), ref.to(device)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu())
        labels.append(y.cpu())
        idx.append(i)
        kls.append(kl_to_reference(logits, ref).cpu())
        ces.append(F.cross_entropy(logits, y, reduction="none").cpu())
    out = {
        "probs": torch.cat(probs).numpy(),
        "labels": torch.cat(labels).numpy(),
        "index": torch.cat(idx).numpy(),
        "kl": torch.cat(kls).numpy(),
        "ce": torch.cat(ces).numpy(),
    }
    return out


def fit(
    model: nn.Module,
    loaders: Dict[str, "torch.utils.data.DataLoader"],
    device: torch.device,
    epochs: int,
    optimizer_name: str = "adamw",
    lr_max: float = 1e-3,
    lr_min: float = 1e-5,
    weight_decay: float = 5e-4,
    class_weights: Optional[Sequence[float]] = None,
    kl_weight: float = 0.0,
    select_best_on: str = "val_acc",
    trainable_params=None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Train with cosine-annealed learning rate and (optionally) a KL regulariser.

    ``loss = weighted_CE + kl_weight * mean KL(P_NAS || P_LR)``.  The weights with the best validation score
    (``val_acc`` or ``val_loss``) are restored at the end.  Returns the per-epoch history.
    """
    model.to(device)
    params = trainable_params if trainable_params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(optimizer_name, params, lr_max, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr_min)
    weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    train_loader, val_loader = loaders["train"], loaders.get("val")

    history: List[Dict[str, float]] = []
    best_state, best_score = None, -np.inf
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = seen = 0
        for x, y, ref, _ in train_loader:
            x, y, ref = x.to(device), y.to(device), ref.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weight)
            if kl_weight > 0:
                loss = loss + kl_weight * kl_to_reference(logits, ref).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / seen, "train_acc": correct / seen}
        if val_loader is not None:
            v = predict(model, val_loader, device)
            row["val_loss"] = float(v["ce"].mean())
            row["val_acc"] = float((v["probs"].argmax(1) == v["labels"]).mean())
            score = row["val_acc"] - 1e-3 * row["val_loss"] if select_best_on == "val_acc" else -row["val_loss"]
            if score > best_score:
                best_score, best_state = score, copy.deepcopy(model.state_dict())
        row["seconds"] = time.time() - t0
        history.append(row)
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            LOG.info("epoch %3d/%d | loss %.4f acc %.4f | val loss %.4f acc %.4f",
                     epoch, epochs, row["train_loss"], row["train_acc"], row.get("val_loss", float("nan")),
                     row.get("val_acc", float("nan")))
        if not np.isfinite(row["train_loss"]):
            raise FloatingPointError("training diverged (non-finite loss)")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
