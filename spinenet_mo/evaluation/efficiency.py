"""Model-complexity reporting: parameters, FLOPs (analytic and measured) and CPU latency."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from ..models.spinenet import count_parameters
from ..search.analytic import count_spec


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_size: int = 224, channels: int = 1, runs: int = 100,
                    warmup: int = 10, device: str = "cpu") -> Dict[str, float]:
    """Single-image forward latency in milliseconds (mean / std / p95)."""
    model = model.to(device).eval()
    x = torch.randn(1, channels, input_size, input_size, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"latency_ms_mean": float(np.mean(times)), "latency_ms_std": float(np.std(times)),
            "latency_ms_p95": float(np.percentile(times, 95))}


def measured_flops(model: torch.nn.Module, input_size: int = 224, channels: int = 1) -> float:
    """FLOPs (2 x multiply-accumulates of matrix / convolution operators) counted by PyTorch's FlopCounterMode, in giga-FLOPs."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.eval().cpu()
    x = torch.randn(1, channels, input_size, input_size)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


def complexity_report(model: torch.nn.Module, spec: dict | None = None, runs: int = 100, warmup: int = 10) -> Dict:
    """Parameter count (PyTorch and analytic), analytic GFLOPs and CPU latency."""
    report = {"params": count_parameters(model), "params_m": count_parameters(model) / 1e6}
    if spec is not None:
        cnt = count_spec(spec)
        report.update({"params_analytic": cnt["params"], "gflops_macs": cnt["macs"] / 1e9,
                       "gflops_2xmacs": 2 * cnt["macs"] / 1e9})
    report["gflops_measured_2xmacs"] = measured_flops(model)
    report.update(measure_latency(model, runs=runs, warmup=warmup))
    return report
