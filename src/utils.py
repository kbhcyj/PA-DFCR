from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Dict, Any
import csv

def calculate_iterative_compression_rate(total_compression: float, num_iterations: int) -> float:
    """Compute per-step compression rate to reach the total compression after N iterations.
    Uses multiplicative scheduling so that (1 - p)^N = 1 - C, hence p = 1 - (1 - C)^(1/N).
    """
    if not (0.0 <= total_compression <= 1.0):
        raise ValueError("total_compression must be in [0, 1]")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be > 0")
    remaining_ratio = 1.0 - total_compression
    per_step = 1.0 - (remaining_ratio ** (1.0 / num_iterations))
    return float(per_step)

def verify_total_from_step(per_step_rate: float, num_iterations: int) -> float:
    """Verify achieved total compression when applying per_step_rate N times."""
    if not (0.0 <= per_step_rate < 1.0):
        raise ValueError("per_step_rate must be in [0, 1)")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be > 0")
    final_remaining = (1.0 - per_step_rate) ** num_iterations
    return 1.0 - final_remaining

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_iterative_csv(rows: Iterable[Dict[str, Any]], out_dir: str, arch: str,
                       dataset: str, total_comp: float, n_iters: int) -> str:
    ensure_dir(out_dir)
    fname = f"iterative_results_{arch}_{dataset}_comp{int(total_comp*100)}_iter{n_iters}_{timestamp()}.csv"
    path = os.path.join(out_dir, fname)
    rows = list(rows)
    if not rows:
        raise ValueError("rows is empty")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path

