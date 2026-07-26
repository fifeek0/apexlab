"""Consistency metrics: lap-time distribution, channel histograms and
point-by-point cross-lap variability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iracing_core import LapData

from .workspace import Workspace

__all__ = [
    "ChannelHistogram",
    "Variability",
    "lap_time_stats",
    "channel_histograms",
    "cross_lap_variability",
]


def lap_time_stats(laps: list[LapData]) -> dict[str, float]:
    times = np.asarray([lap.lap_time for lap in laps if lap.lap_time is not None])
    if len(times) == 0:
        return {"count": 0}
    return {
        "count": int(len(times)),
        "best": float(times.min()),
        "worst": float(times.max()),
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "std": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
    }


@dataclass(frozen=True)
class ChannelHistogram:
    channel: str
    edges: np.ndarray  # shape (bins + 1,)
    counts: np.ndarray  # shape (n_laps, bins)
    lap_labels: list[str]


def channel_histograms(workspace: Workspace, channel: str, bins: int = 20) -> ChannelHistogram:
    """Per-lap histograms of one channel over the common range."""
    arrays = [al.channels[channel].astype(float) for al in workspace.aligned.laps]
    lo = min(a.min() for a in arrays)
    hi = max(a.max() for a in arrays)
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.stack([np.histogram(a, bins=edges)[0] for a in arrays])
    return ChannelHistogram(
        channel=channel, edges=edges, counts=counts, lap_labels=workspace.lap_labels()
    )


@dataclass(frozen=True)
class Variability:
    """Standard deviation across laps at every grid point: where is the
    driving repeatable and where does it scatter?"""

    speed_std: np.ndarray
    throttle_std: np.ndarray
    brake_std: np.ndarray
    steer_std: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "speed_std_mean_ms": float(self.speed_std.mean()),
            "throttle_std_mean": float(self.throttle_std.mean()),
            "brake_std_mean": float(self.brake_std.mean()),
            "steer_std_mean_rad": float(self.steer_std.mean()),
        }


def cross_lap_variability(workspace: Workspace) -> Variability:
    def std_of(channel: str) -> np.ndarray:
        stacks = [
            al.channels[channel].astype(float)
            for al in workspace.aligned.laps
            if channel in al.channels
        ]
        if len(stacks) < 2:
            return np.zeros_like(workspace.grid)
        return np.std(np.stack(stacks), axis=0)

    return Variability(
        speed_std=std_of("Speed"),
        throttle_std=std_of("Throttle"),
        brake_std=std_of("Brake"),
        steer_std=std_of("SteeringWheelAngle"),
    )
