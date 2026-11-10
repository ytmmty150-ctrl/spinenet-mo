"""Genome -> architecture specification.

The specification is a plain-python description of the network.  It is the single
source of truth for both the analytic cost model (:mod:`spinenet_mo.search.analytic`)
and the PyTorch builder (:mod:`spinenet_mo.models.spinenet`), so the objectives
``f_2`` (FLOPs) and the reported parameter counts always agree with the network that
is actually trained.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .space import genome_to_dict

NUM_CLASSES = 3
INPUT_CHANNELS = 1
INPUT_SIZE = 224
STEM_CHANNELS = 32
STEM_KERNEL = 7
DROP_NUM = 2  # dropout layers in the head (Table 2)
LIN_NUM = 2   # hidden fully-connected layers in the head (Table 2)
FLATTEN_GRID = 4  # spatial grid kept when the head does not use global average pooling


def eca_kernel_size(channels: int, psi: int = 2, b: int = 1) -> int:
    """Adaptive ECA kernel size :math:`k = |\\log_2 C/\\psi + b/\\psi|_{odd}` (Eq. 8)."""
    k = int(abs(math.log2(channels) / psi + b / psi))
    return k if k % 2 == 1 else k + 1


def _node_op(conv_type: str, cin: int, cout: int, k: int, stride: int) -> Dict[str, Any]:
    if conv_type == "standard":
        return {"op": "conv", "cin": cin, "cout": cout, "k": k, "stride": stride}
    if conv_type == "depthwise":
        return {"op": "dwsep", "cin": cin, "cout": cout, "k": k, "stride": stride}
    if conv_type == "ghost":
        if cout % 2:
            raise ValueError("GhostConv requires an even number of output channels")
        return {"op": "ghost", "cin": cin, "cout": cout, "k": k, "stride": stride, "ratio": 2, "dw_k": 3}
    raise ValueError(conv_type)


def _attention(kind: str, channels: int) -> List[Dict[str, Any]]:
    if kind == "none":
        return []
    if kind == "eca":
        return [{"op": "eca", "c": channels, "k": eca_kernel_size(channels)}]
    if kind == "cbam":
        return [{"op": "cbam", "c": channels, "reduction": 16, "spatial_k": 7}]
    raise ValueError(kind)


def build_spec(genome: Sequence[int], num_blocks: int) -> Dict[str, Any]:
    """Decode ``genome`` into a layer-by-layer specification.

    Each macro-block is expanded into ``max(1, graph_nodes // L)`` computational nodes
    (conv -> norm -> activation) followed by the optional attention module, matching
    the footnote of Table 2 ("each macro-block is itself expanded into multiple layers").
    """
    d = genome_to_dict(genome, num_blocks)
    act = d["activation"]
    nodes_per_block = max(1, int(d["graph_nodes"]) // num_blocks)

    stem = [
        {"op": "conv", "cin": INPUT_CHANNELS, "cout": STEM_CHANNELS, "k": STEM_KERNEL, "stride": 2},
        {"op": "norm", "kind": "batch", "c": STEM_CHANNELS},
        {"op": "act", "kind": act},
    ]
    blocks: List[Dict[str, Any]] = []
    cin = STEM_CHANNELS
    for b in d["blocks"]:
        cout, k, stride = int(b["filters"]), int(b["kernel"]), int(b["stride"])
        layers: List[Dict[str, Any]] = []
        c_cur = cin
        for node in range(nodes_per_block):
            layers.append(_node_op(b["conv_type"], c_cur, cout, k, stride if node == 0 else 1))
            layers.append({"op": "norm", "kind": b["norm"], "c": cout})
            layers.append({"op": "act", "kind": act})
            c_cur = cout
        layers += _attention(d["attention"], cout)
        residual = None
        if b["residual"]:
            residual = {"cin": cin, "cout": cout, "stride": stride, "project": (cin != cout or stride != 1)}
        blocks.append({"layers": layers, "residual": residual, "cin": cin, "cout": cout})
        cin = cout

    pooling = d["pooling"]
    head: List[Dict[str, Any]] = []
    if pooling == "global_avg":
        head.extend([{"op": "gap"}, {"op": "flatten"}])
        feat = cin
    else:
        head.append({"op": "adaptive_pool", "kind": pooling, "size": FLATTEN_GRID})
        head.append({"op": "flatten"})
        feat = cin * FLATTEN_GRID * FLATTEN_GRID
    width = int(d["fc_units"])
    for i in range(LIN_NUM):
        if i < DROP_NUM:
            head.append({"op": "dropout", "p": float(d["dropout"])})
        head.append({"op": "linear", "in": feat if i == 0 else width, "out": width})
        head.append({"op": "act", "kind": act})
    for _ in range(max(0, DROP_NUM - LIN_NUM)):
        head.append({"op": "dropout", "p": float(d["dropout"])})
    head.append({"op": "linear", "in": width, "out": NUM_CLASSES})

    return {
        "input": {"channels": INPUT_CHANNELS, "size": INPUT_SIZE},
        "stem": stem,
        "blocks": blocks,
        "head": head,
        "train_hparams": {"optimizer": d["optimizer"], "lr": float(d["lr"])},
        "description": d,
    }
