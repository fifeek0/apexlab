"""Track-map geometry: GPS (or velocity/yaw) channels → local metres."""

from __future__ import annotations

import numpy as np

__all__ = ["latlon_to_local_xy", "shape_from_aligned_lap"]

_M_PER_DEG_LAT = 111_320.0


def latlon_to_local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    origin: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection around ``origin`` (lat0, lon0), defaulting
    to the trace centroid.

    Plenty accurate for a few km of racetrack (sub-centimetre distortion),
    and keeps true metric proportions for the map view. Pass a shared
    ``origin`` to project several laps into one frame (driving-line
    comparison).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0, lon0 = origin if origin is not None else (float(lat.mean()), float(lon.mean()))
    x = (lon - lon0) * _M_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * _M_PER_DEG_LAT
    return x, y


def shape_from_aligned_lap(
    channels: dict[str, np.ndarray],
    time_at: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Track outline (x, y in metres) from one distance-aligned lap.

    Prefers GPS (``Lat``/``Lon``); falls back to dead reckoning from
    ``Speed`` + ``Yaw`` when GPS channels are absent.
    """
    if "Lat" in channels and "Lon" in channels:
        return latlon_to_local_xy(channels["Lat"], channels["Lon"])

    if "Speed" in channels and "Yaw" in channels:
        speed = channels["Speed"]
        yaw = channels["Yaw"]
        dt = np.diff(time_at, prepend=time_at[0])
        x = np.cumsum(speed * np.cos(yaw) * dt)
        y = np.cumsum(speed * np.sin(yaw) * dt)
        return x - x.mean(), y - y.mean()

    return None
