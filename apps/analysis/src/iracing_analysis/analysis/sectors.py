"""Sector and mini-sector timing on distance-aligned laps.

Sector times come from the elapsed-time-at-distance curve (``time_at``),
which is anchored to exact S/F crossings — so a lap's sector times sum to
its lap time to sub-millisecond precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .workspace import Workspace

__all__ = [
    "SectorTable",
    "official_sector_boundaries",
    "minisector_boundaries",
    "sector_times",
]


@dataclass
class SectorTable:
    """Per-lap, per-sector times. ``boundaries_m[k]`` starts sector ``k``."""

    boundaries_m: np.ndarray  # shape (n_sectors,), first must be 0.0
    times: np.ndarray  # shape (n_laps, n_sectors), seconds
    lap_labels: list[str]

    @property
    def n_sectors(self) -> int:
        return len(self.boundaries_m)

    @property
    def best_lap(self) -> np.ndarray:
        """Index of the fastest lap in each sector — shape (n_sectors,)."""
        return np.argmin(self.times, axis=0)

    @property
    def best_times(self) -> np.ndarray:
        """Fastest time per sector — the mini-sector basis of the theoretical best."""
        return np.min(self.times, axis=0)


def _track_length(workspace: Workspace) -> float:
    meta = workspace.laps[0].meta
    if meta and meta.track_length_m:
        # prefer telemetry-derived length so boundaries live on the grid
        pass
    grid_end = float(workspace.grid[-1]) + workspace.spacing
    if meta and meta.track_length_m and abs(meta.track_length_m - grid_end) < 60.0:
        return float(meta.track_length_m)
    return grid_end


def official_sector_boundaries(workspace: Workspace) -> np.ndarray:
    """iRacing's official sector starts (from SplitTimeInfo) in metres."""
    meta = workspace.laps[0].meta
    track_len = _track_length(workspace)
    pcts = meta.sector_starts_pct if meta and meta.sector_starts_pct else [0.0]
    return np.asarray([p * track_len for p in pcts], dtype=float)


def minisector_boundaries(workspace: Workspace, count: int = 20) -> np.ndarray:
    """``count`` equal-length mini-sectors starting at S/F."""
    if count < 2:
        raise ValueError("need at least 2 mini-sectors")
    track_len = _track_length(workspace)
    return np.linspace(0.0, track_len, count + 1)[:-1]


def sector_times(workspace: Workspace, boundaries_m: np.ndarray) -> SectorTable:
    """Time spent by each lap in each sector.

    The final sector closes at the S/F line using the exact lap time; laps
    without a lap time (partials) get NaN in sectors they didn't complete.
    """
    boundaries = np.asarray(boundaries_m, dtype=float)
    if boundaries[0] != 0.0:
        raise ValueError("first sector must start at 0 m (the S/F line)")
    grid = workspace.grid
    n_laps = len(workspace.aligned.laps)
    times = np.full((n_laps, len(boundaries)), np.nan)

    for i, al in enumerate(workspace.aligned.laps):
        t_at = np.interp(boundaries, grid, al.time_at)
        lap_time = al.lap.lap_time
        end_time = lap_time if lap_time is not None else float(al.time_at[-1])
        edges = np.append(t_at, end_time)
        times[i] = np.diff(edges)

    return SectorTable(
        boundaries_m=boundaries,
        times=times,
        lap_labels=workspace.lap_labels(),
    )
