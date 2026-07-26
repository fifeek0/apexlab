"""Theoretical best lap from best mini-sector times.

IMPORTANT CAVEAT: this is an *optimistic upper bound* on improvement. It
stitches together the best mini-sectors from different laps, which may not
be simultaneously achievable (a mini-sector gained by braking later steals
speed from the next one; tyre/fuel states differ between laps). Treat the
gap as "spread of your own performance", not as a directly reachable time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sectors import minisector_boundaries, sector_times
from .workspace import Workspace

__all__ = ["TheoreticalBest", "theoretical_best"]


@dataclass(frozen=True)
class TheoreticalBest:
    minisector_count: int
    boundaries_m: np.ndarray
    best_times: np.ndarray  # per mini-sector, seconds
    contributors: np.ndarray  # lap index that provided each best mini-sector
    total: float
    actual_best: float
    gap: float  # actual_best - total (>= 0)

    def label(self) -> str:
        mins, secs = divmod(self.total, 60.0)
        return f"{int(mins)}:{secs:06.3f}"


def theoretical_best(workspace: Workspace, minisectors: int = 20) -> TheoreticalBest:
    """Sum of best mini-sector times across the selected (complete) laps."""
    table = sector_times(workspace, minisector_boundaries(workspace, minisectors))
    complete = [
        i for i, al in enumerate(workspace.aligned.laps) if al.lap.lap_time is not None
    ]
    if not complete:
        raise ValueError("theoretical best needs at least one complete lap")
    times = table.times[complete]

    best_idx_local = np.nanargmin(times, axis=0)
    best_times = times[best_idx_local, np.arange(times.shape[1])]
    contributors = np.asarray([complete[i] for i in best_idx_local])
    actual_best = min(workspace.aligned.laps[i].lap.lap_time for i in complete)
    total = float(best_times.sum())
    return TheoreticalBest(
        minisector_count=minisectors,
        boundaries_m=table.boundaries_m,
        best_times=best_times,
        contributors=contributors,
        total=total,
        actual_best=float(actual_best),
        gap=float(actual_best - total),
    )
