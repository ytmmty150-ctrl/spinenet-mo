"""Analytic parameter / FLOP counter (objective ``f_2``, Eq. 3).

Cost is computed directly from the architecture specification, without building or
running the network.  Conventions (documented, and checked against PyTorch in the tests):

* "FLOPs" are **multiply-accumulates** (MACs) of convolution, depth-wise, ghost,
  attention and linear layers for a single ``224 x 224 x 1`` image.  Set
  ``macs=False`` to report 2 x MACs.
* Normalisation, activation, pooling and element-wise additions are not counted.
* Parameters are trainable weights and biases (normalisation affine parameters
  included, running statistics excluded).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

Cost = Tuple[int, int]  # (params, macs)


def _out(size: int, k: int, stride: int) -> int:
    pad = k // 2
    return (size + 2 * pad - k) // stride + 1


def _layer_cost(layer: Dict[str, Any], c: int, h: int, w: int) -> Tuple[int, int, int, int, int]:
    """Return (params, macs, c_out, h_out, w_out) for one layer."""
    op = layer["op"]
    if op == "conv":
        ho, wo = _out(h, layer["k"], layer["stride"]), _out(w, layer["k"], layer["stride"])
        p = layer["cout"] * layer["cin"] * layer["k"] ** 2
        return p, p * ho * wo, layer["cout"], ho, wo
    if op == "dwsep":
        ho, wo = _out(h, layer["k"], layer["stride"]), _out(w, layer["k"], layer["stride"])
        p_dw = layer["cin"] * layer["k"] ** 2
        p_pw = layer["cin"] * layer["cout"]
        return p_dw + p_pw, (p_dw + p_pw) * ho * wo, layer["cout"], ho, wo
    if op == "ghost":
        ho, wo = _out(h, layer["k"], layer["stride"]), _out(w, layer["k"], layer["stride"])
        m = layer["cout"] // layer["ratio"]
        p_primary = m * layer["cin"] * layer["k"] ** 2
        p_cheap = m * layer["dw_k"] ** 2 * (layer["ratio"] - 1)
        return p_primary + p_cheap, (p_primary + p_cheap) * ho * wo, layer["cout"], ho, wo
    if op == "norm":
        return 2 * layer["c"], 0, c, h, w
    if op in ("act", "dropout", "flatten"):
        if op == "flatten":
            return 0, 0, c * h * w, 1, 1
        return 0, 0, c, h, w
    if op == "eca":
        return layer["k"], layer["k"] * layer["c"], c, h, w
    if op == "cbam":
        ch = layer["c"]
        hidden = max(1, ch // layer["reduction"])
        p_mlp = ch * hidden + hidden + hidden * ch + ch  # shared MLP with biases
        p_sp = 2 * layer["spatial_k"] ** 2 + 1             # 2-channel -> 1-channel spatial conv with bias
        macs = 2 * (ch * hidden + hidden * ch) + p_sp * h * w  # MLP applied to avg- and max-pooled descriptors
        return p_mlp + p_sp, macs, c, h, w
    if op == "gap":
        return 0, 0, c, 1, 1
    if op == "adaptive_pool":
        return 0, 0, c, layer["size"], layer["size"]
    if op == "linear":
        p = layer["in"] * layer["out"] + layer["out"]
        return p, layer["in"] * layer["out"], layer["out"], 1, 1
    raise ValueError(f"Unknown op: {op}")


def count_spec(spec: Dict[str, Any]) -> Dict[str, float]:
    """Parameters and MACs of a specification (see module docstring)."""
    params = macs = 0
    c, h, w = spec["input"]["channels"], spec["input"]["size"], spec["input"]["size"]

    def run(layers, c, h, w):
        nonlocal params, macs
        for layer in layers:
            p, m, c, h, w = _layer_cost(layer, c, h, w)
            params += p
            macs += m
        return c, h, w

    c, h, w = run(spec["stem"], c, h, w)
    for block in spec["blocks"]:
        c_in, h_in, w_in = c, h, w
        c, h, w = run(block["layers"], c, h, w)
        res = block["residual"]
        if res and res["project"]:  # 1x1 projection conv + batch norm on the shortcut
            ho, wo = _out(h_in, 1, res["stride"]), _out(w_in, 1, res["stride"])
            p = res["cout"] * res["cin"] + 2 * res["cout"]
            params += p
            macs += res["cout"] * res["cin"] * ho * wo
    run(spec["head"], c, h, w)
    return {"params": int(params), "macs": int(macs)}


def cost_of_genome(genome, num_blocks: int, macs: bool = True) -> Dict[str, float]:
    """Convenience wrapper: ``{"params_m": ..., "gflops": ...}`` for a genome."""
    from .arch import build_spec

    spec = build_spec(genome, num_blocks)
    cnt = count_spec(spec)
    factor = 1 if macs else 2
    return {"params": cnt["params"], "params_m": cnt["params"] / 1e6, "gflops": factor * cnt["macs"] / 1e9}
