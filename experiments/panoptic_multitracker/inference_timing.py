"""Small synchronized timing helper shared by the experiment runners."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    """Run a callable with explicit device synchronization and wall timing."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    result = None
    for _ in range(warmup):
        result = function()
        _synchronize(device)
        del result

    samples = []
    peak_allocated = []
    incremental_peak = []
    for _ in range(repeats):
        _synchronize(device)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = 0
        started = time.perf_counter()
        result = function()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            peak_allocated.append(peak / (1024**2))
            incremental_peak.append(max(peak - baseline, 0) / (1024**2))
        if len(samples) < repeats:
            del result

    values = np.asarray(samples, dtype=np.float64)
    timing = {
        "warmup": warmup,
        "repeats": repeats,
        "samples_seconds": values.tolist(),
        "median_seconds": float(np.median(values)),
        "p25_seconds": float(np.percentile(values, 25)),
        "p75_seconds": float(np.percentile(values, 75)),
        "peak_allocated_mib": float(max(peak_allocated)) if peak_allocated else None,
        "incremental_peak_mib": float(max(incremental_peak)) if incremental_peak else None,
    }
    return result, timing
