"""GG / friction-circle data: combined lateral+longitudinal grip usage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iracing_core import AlignedLap

__all__ = ["GGPoints", "gg_points", "gg_stats"]

G = 9.80665  # m/s^2


@dataclass(frozen=True)
class GGPoints:
    """Accelerations in g for one lap (distance-aligned)."""

    lat_g: np.ndarray
    long_g: np.ndarray

    @property
    def combined_g(self) -> np.ndarray:
        return np.hypot(self.lat_g, self.long_g)


def gg_points(lap: AlignedLap) -> GGPoints:
    lat = lap.channels.get("LatAccel")
    lon = lap.channels.get("LongAccel")
    if lat is None or lon is None:
        raise KeyError("LatAccel/LongAccel channels not loaded for this lap")
    return GGPoints(lat_g=np.asarray(lat, float) / G, long_g=np.asarray(lon, float) / G)


def gg_stats(gg: GGPoints) -> dict[str, float]:
    """Headline grip-usage numbers for reports and the AI summary."""
    combined = gg.combined_g
    return {
        "max_lat_g": float(np.abs(gg.lat_g).max()),
        "max_accel_g": float(gg.long_g.max()),
        "max_brake_g": float(-gg.long_g.min()),
        "max_combined_g": float(combined.max()),
        "combined_p95_g": float(np.percentile(combined, 95)),
        "combined_mean_g": float(combined.mean()),
    }
