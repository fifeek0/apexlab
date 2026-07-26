"""Distance-based lap alignment and delta-time computation.

Laps of different durations are resampled onto a common LapDist grid so
they align *spatially*. Elapsed lap time as a function of distance is
interpolated from ``SessionTime`` and anchored to the exact S/F crossing
instants stored on :class:`~iracing_core.models.LapData`, which makes the
cumulative delta at the finish line agree with the lap-time difference to
well under a millisecond.

Used by both the analysis app (delta graph, trace overlays, corner
attribution) and the overlay's live delta engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import LapData

__all__ = ["AlignedLap", "AlignedLapSet", "align_lap", "align_laps", "delta_time"]

#: channels that are integral/categorical — resampled with nearest-neighbour
INTEGER_CHANNELS: frozenset[str] = frozenset(
    {"Gear", "Lap", "PlayerTrackSurface", "OnPitRoad", "SessionTick"}
)

#: channels that make no sense on a distance grid
_SKIP_CHANNELS: frozenset[str] = frozenset({"LapDist", "LapDistPct"})


@dataclass
class AlignedLap:
    """One lap resampled onto a common distance grid."""

    lap: LapData
    grid: np.ndarray  # distance from S/F, m
    channels: dict[str, np.ndarray] = field(repr=False)
    time_at: np.ndarray = field(repr=False)  # elapsed lap time at each grid point, s

    def label(self) -> str:
        return self.lap.label()


@dataclass
class AlignedLapSet:
    """Several laps of the same track on one shared distance grid."""

    grid: np.ndarray
    laps: list[AlignedLap]

    def deltas_to(self, reference: int | AlignedLap) -> list[np.ndarray]:
        """Delta-time of every lap against ``reference`` (index or lap)."""
        ref = self.laps[reference] if isinstance(reference, int) else reference
        return [delta_time(lap, ref) for lap in self.laps]


def _monotonic_distance(dist: np.ndarray) -> np.ndarray:
    """LapDist should already be monotonic within a lap; enforce it against
    sensor jitter so np.interp gets a valid x-axis."""
    return np.maximum.accumulate(dist.astype(float))


def _elapsed_time(lap: LapData, distance_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(distance, elapsed seconds) support points, anchored at the exact
    S/F crossings when known."""
    dist = _monotonic_distance(lap.channel("LapDist")) * distance_scale
    time = lap.channel("SessionTime").astype(float)

    t0 = lap.crossing_start_time if lap.crossing_start_time is not None else float(time[0])
    elapsed = time - t0

    xs = [dist]
    ys = [elapsed]
    if lap.crossing_start_time is not None and dist[0] > 0.0:
        xs.insert(0, np.array([0.0]))
        ys.insert(0, np.array([0.0]))
    if lap.lap_time is not None:
        track_len = _lap_length(lap) * distance_scale
        if track_len > dist[-1]:
            xs.append(np.array([track_len]))
            ys.append(np.array([lap.lap_time]))
    return np.concatenate(xs), np.concatenate(ys)


def _lap_length(lap: LapData) -> float:
    """Best estimate of the full lap length in metres."""
    dist = lap.channel("LapDist")
    pct = lap.channel("LapDistPct")
    tail_pct = float(pct[-1])
    if tail_pct > 0.5:  # extrapolate from how far around the lap we got
        est = float(dist[-1]) / tail_pct
    else:
        est = float(dist[-1])
    if lap.meta and lap.meta.track_length_m:
        # YAML track length is rounded to 10 m; prefer telemetry when close
        if abs(est - lap.meta.track_length_m) > 50.0:
            return float(lap.meta.track_length_m)
    return est


def align_lap(
    lap: LapData,
    grid: np.ndarray,
    channels: list[str] | None = None,
    distance_scale: float = 1.0,
) -> AlignedLap:
    """Resample one lap's channels onto ``grid`` (metres from S/F).

    ``distance_scale`` rescales the lap's own distance axis before
    resampling — used by :func:`align_laps` to bring laps whose distance
    reconstructions disagree slightly (e.g. Garage 61 CSV imports, where
    length is estimated from each lap's own speed integral) onto one
    consistent metre scale.
    """
    dist = _monotonic_distance(lap.channel("LapDist")) * distance_scale
    names = channels if channels is not None else [
        name for name in lap.channels if name not in _SKIP_CHANNELS and name != "SessionTime"
    ]

    resampled: dict[str, np.ndarray] = {}
    nearest_idx: np.ndarray | None = None
    for name in names:
        values = lap.channel(name)
        if name in INTEGER_CHANNELS or values.dtype == np.bool_:
            if nearest_idx is None:
                nearest_idx = np.clip(
                    np.searchsorted(dist, grid, side="left"), 0, len(dist) - 1
                )
            resampled[name] = values[nearest_idx]
        else:
            resampled[name] = np.interp(grid, dist, values.astype(float))

    t_dist, t_elapsed = _elapsed_time(lap, distance_scale)
    time_at = np.interp(grid, t_dist, t_elapsed)
    return AlignedLap(lap=lap, grid=grid, channels=resampled, time_at=time_at)


def align_laps(
    laps: list[LapData],
    spacing: float = 1.0,
    channels: list[str] | None = None,
) -> AlignedLapSet:
    """Align several laps (same track!) onto one common distance grid.

    Each lap's distance axis is rescaled to the *median* estimated lap
    length, so laps whose own length reconstructions disagree by a fraction
    of a percent (different driven lines in CSV imports) still align; for
    laps parsed from the same sim (.ibt) the estimates coincide and the
    rescale is a no-op. The grid spans ``[0, min(covered distance))`` at
    ``spacing`` metres so every lap has real data along the whole grid.
    """
    if not laps:
        raise ValueError("no laps to align")
    lengths = [_lap_length(lap) for lap in laps]
    target = float(np.median(lengths))
    scales = [target / length if length > 0 else 1.0 for length in lengths]

    ends = []
    for lap, scale in zip(laps, scales):
        if lap.lap_time is not None:  # complete lap: time anchors cover the full length
            ends.append(target)
        else:  # partial lap: only as far as it actually drove
            ends.append(float(_monotonic_distance(lap.channel("LapDist"))[-1]) * scale)
    grid = np.arange(0.0, min(ends), spacing)
    return AlignedLapSet(
        grid=grid,
        laps=[
            align_lap(lap, grid, channels, distance_scale=scale)
            for lap, scale in zip(laps, scales)
        ],
    )


def delta_time(target: AlignedLap, reference: AlignedLap) -> np.ndarray:
    """Cumulative time gained(+)/lost(-)... by convention here: positive
    means ``target`` is *slower* (has lost time) than ``reference`` at that
    point around the lap. Both laps must share the same grid."""
    if target.grid is not reference.grid and not np.array_equal(target.grid, reference.grid):
        raise ValueError("laps are not on the same distance grid")
    return target.time_at - reference.time_at
