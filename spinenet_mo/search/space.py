"""Architecture search space and integer genome (paper Table 1, Section 3.3).

A candidate is a fixed-length integer vector ``v`` of dimension ``d = 6 L + G``:

* ``L`` macro-blocks, each described by six genes
  ``[conv_type, filters, kernel, stride, residual, norm]`` (the paper's
  ``[t, c, k, s, r, l]`` sub-vector), and
* ``G = 8`` global genes (activation, FC units, dropout, pooling, attention,
  graph nodes, optimizer, learning rate).

Every gene stores an *index* into the candidate list of the corresponding search
dimension, so uniform sampling, crossover and mutation are trivial and every
genome decodes to a valid architecture.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np

#: Candidate values of every searchable parameter (Table 1 of the manuscript).
SEARCH_SPACE: "OrderedDict[str, list]" = OrderedDict(
    filters=[16, 32, 64, 128],
    kernel=[1, 3, 5, 7],
    activation=["relu", "leakyrelu", "gelu", "swish"],
    fc_units=[256, 512, 1024],
    graph_nodes=[5, 8, 12, 16],
    dropout=[0.1, 0.3, 0.5],
    pooling=["max", "avg", "global_avg"],
    norm=["batch", "layer", "group"],
    residual=[True, False],
    optimizer=["sgd", "adam", "adamw"],
    lr=[0.01, 0.001, 0.0001],
    conv_type=["standard", "depthwise", "ghost"],
    attention=["none", "eca", "cbam"],
)

#: Extra (non-searched) block gene: stride 1 or 2.
STRIDES = [1, 2]

BLOCK_GENES = ["conv_type", "filters", "kernel", "stride", "residual", "norm"]
GLOBAL_GENES = ["activation", "fc_units", "dropout", "pooling", "attention", "graph_nodes", "optimizer", "lr"]


def _choices(name: str) -> list:
    return STRIDES if name == "stride" else SEARCH_SPACE[name]


def genome_length(num_blocks: int) -> int:
    return len(BLOCK_GENES) * num_blocks + len(GLOBAL_GENES)


def cardinalities(num_blocks: int) -> np.ndarray:
    """Number of admissible values for every gene position."""
    card = []
    for _ in range(num_blocks):
        card += [len(_choices(g)) for g in BLOCK_GENES]
    card += [len(_choices(g)) for g in GLOBAL_GENES]
    return np.asarray(card, dtype=np.int64)


def random_genome(num_blocks: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray([rng.integers(0, c) for c in cardinalities(num_blocks)], dtype=np.int64)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, prob: float = 0.6):
    """Uniform crossover applied with probability ``prob`` (otherwise copies)."""
    if rng.random() >= prob:
        return a.copy(), b.copy()
    mask = rng.random(a.shape[0]) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def mutate(genome: np.ndarray, card: np.ndarray, rng: np.random.Generator, prob: float | None = None) -> np.ndarray:
    """Gene-wise re-sampling mutation (the integer analogue of bit-flip), ``p_m = 1/d`` by default.

    ``card`` holds the number of admissible values of every gene (see :func:`cardinalities`).
    A mutated gene always receives a *different* value.
    """
    p = (1.0 / len(genome)) if prob is None else prob
    out = genome.copy()
    flip = rng.random(len(genome)) < p
    for i in np.flatnonzero(flip):
        if card[i] > 1:
            new = rng.integers(0, card[i] - 1)
            out[i] = new if new < out[i] else new + 1
    return out


def genome_to_dict(genome: Sequence[int], num_blocks: int) -> Dict[str, object]:
    """Human readable decoding of a genome."""
    genome = [int(g) for g in genome]
    blocks: List[Dict[str, object]] = []
    pos = 0
    for _ in range(num_blocks):
        block = {}
        for name in BLOCK_GENES:
            block[name] = _choices(name)[genome[pos]]
            pos += 1
        blocks.append(block)
    glob = {}
    for name in GLOBAL_GENES:
        glob[name] = _choices(name)[genome[pos]]
        pos += 1
    return {"blocks": blocks, **glob}


def genome_key(genome: Sequence[int]) -> tuple:
    return tuple(int(g) for g in genome)


def with_overrides(genome: Sequence[int], num_blocks: int, **values) -> np.ndarray:
    """Return a copy of ``genome`` with named genes set to the given *values*.

    Block-level names (``conv_type``, ``filters`` ...) are applied to **every** block;
    global names (``attention``, ``pooling`` ...) to the global gene.  ``conv_type``
    additionally supports ``replace={"ghost": "standard"}``-style substitution via
    :func:`replace_conv_type`.
    """
    out = np.asarray(genome, dtype=np.int64).copy()
    nb = len(BLOCK_GENES)
    for name, value in values.items():
        choices = _choices(name)
        idx = choices.index(value)
        if name in BLOCK_GENES:
            j = BLOCK_GENES.index(name)
            for b in range(num_blocks):
                out[b * nb + j] = idx
        elif name in GLOBAL_GENES:
            out[nb * num_blocks + GLOBAL_GENES.index(name)] = idx
        else:
            raise KeyError(name)
    return out


def replace_conv_type(genome: Sequence[int], num_blocks: int, old: str, new: str) -> np.ndarray:
    """Replace one convolution type by another in every block (used for the 'w/o GhostConv' ablation)."""
    out = np.asarray(genome, dtype=np.int64).copy()
    nb = len(BLOCK_GENES)
    j = BLOCK_GENES.index("conv_type")
    types = SEARCH_SPACE["conv_type"]
    for b in range(num_blocks):
        if types[out[b * nb + j]] == old:
            out[b * nb + j] = types.index(new)
    return out
