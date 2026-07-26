"""Real-time comparison of the current lap against a reference lap.

The brain of the Bloops-style overlay: feed live samples, get back the
current time delta to the reference (anchored at the S/F crossing exactly
like the post-session delta engine), the speed gap, the reference's
gear/throttle/brake at your position, the distance to the next braking
zone (for audio cues) and a window of upcoming reference inputs (for the
input-trace preview). UI-agnostic and driven sample-by-sample, so it works
identically against the live sim, a replay, or a test stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import align_lap
from .models import LapData

__all__ = ["LiveComparison", "LiveState", "ReferenceWindow"]

#: minimum brake application that counts as a braking zone
_BRAKE_ON = 0.10
#: merge zones closer than this (m)
_ZONE_GAP_M = 30.0


@dataclass(frozen=True)
class LiveState:
    """Everything the overlay needs at one telemetry tick."""

    lap_dist_m: float
    lap_pct: float
    time_delta: float  # + = behind the reference (losing)
    speed_delta_kmh: float  # + = faster than the reference here
    my_gear: int
    my_throttle: float
    my_brake: float
    ref_gear: int
    ref_throttle: float
    ref_brake: float
    next_braking_m: float | None  # distance to the next reference braking zone
    lap_number: int


@dataclass(frozen=True)
class ReferenceWindow:
    """Reference inputs around the current position (offsets in metres)."""

    offsets_m: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    speed: np.ndarray


class LiveComparison:
    def __init__(self, reference: LapData, spacing: float = 2.0):
        if reference.lap_time is None:
            raise ValueError("reference lap has no lap time")
        self.reference = reference
        length = float(np.max(reference.channel("LapDist")))
        pct_tail = float(reference.channel("LapDistPct")[-1])
        if pct_tail > 0.5:
            length = length / pct_tail
        self.track_length_m = length

        grid = np.arange(0.0, length, spacing)
        self._ref = align_lap(reference, grid)
        self._grid = grid
        self._spacing = spacing

        brake = self._ref.channels.get("Brake")
        self.braking_zones_m: list[float] = (
            self._find_braking_zones(brake) if brake is not None else []
        )

        self._t_lap_start: float | None = None
        self._prev: dict | None = None
        self._last_dist: float = 0.0

    # -- reference geometry ------------------------------------------------

    def _find_braking_zones(self, brake: np.ndarray) -> list[float]:
        on = np.asarray(brake, float) > _BRAKE_ON
        starts: list[float] = []
        gap = int(_ZONE_GAP_M / self._spacing)
        i = 0
        while i < len(on):
            if on[i]:
                starts.append(float(self._grid[i]))
                j = i
                while j < len(on) and (on[j] or (j - i) < gap or on[max(0, j - gap) : j].any()):
                    j += 1
                i = j
            else:
                i += 1
        return starts

    # -- live feed ------------------------------------------------------------

    def feed(self, sample: dict) -> LiveState | None:
        """Consume one live sample; returns the comparison state once the
        lap is anchored (first S/F crossing or an on-line start)."""
        t = float(sample["SessionTime"])
        pct = float(sample["LapDistPct"])
        dist = float(sample.get("LapDist", pct * self.track_length_m))

        if self._t_lap_start is None and pct < 0.005:
            self._t_lap_start = t
        elif self._prev is not None:
            prev_pct = float(self._prev["LapDistPct"])
            if pct < prev_pct - 0.5:  # S/F crossing
                to_run = 1.0 - prev_pct
                frac = to_run / max(to_run + pct, 1e-12)
                t_prev = float(self._prev["SessionTime"])
                self._t_lap_start = t_prev + frac * (t - t_prev)

        self._prev = sample
        self._last_dist = dist
        if self._t_lap_start is None:
            return None

        elapsed = t - self._t_lap_start
        idx = int(np.clip(np.searchsorted(self._grid, dist), 0, len(self._grid) - 1))
        ref_time = float(self._ref.time_at[idx])
        ref_speed = float(self._ref.channels["Speed"][idx])
        my_speed = float(sample.get("Speed", ref_speed))

        upcoming = [z for z in self.braking_zones_m if z > dist]
        next_zone = (upcoming[0] - dist) if upcoming else (
            (self.braking_zones_m[0] + self.track_length_m - dist)
            if self.braking_zones_m else None
        )

        return LiveState(
            lap_dist_m=dist,
            lap_pct=pct,
            time_delta=elapsed - ref_time,
            speed_delta_kmh=(my_speed - ref_speed) * 3.6,
            my_gear=int(sample.get("Gear", 0)),
            my_throttle=float(sample.get("Throttle", 0.0)),
            my_brake=float(sample.get("Brake", 0.0)),
            ref_gear=int(self._ref.channels.get("Gear", np.zeros(1))[idx]),
            ref_throttle=float(self._ref.channels.get("Throttle", np.zeros(len(self._grid)))[idx]),
            ref_brake=float(self._ref.channels.get("Brake", np.zeros(len(self._grid)))[idx]),
            next_braking_m=next_zone,
            lap_number=int(sample.get("Lap", 0)),
        )

    # -- overlay helpers ----------------------------------------------------------

    def reference_window(
        self, behind_m: float = 250.0, ahead_m: float = 150.0
    ) -> ReferenceWindow | None:
        """Reference inputs around the current position for the trace strip."""
        if self._prev is None:
            return None
        dist = self._last_dist
        lo = int(np.clip(np.ceil((dist - behind_m) / self._spacing), 0, len(self._grid) - 1))
        hi = int(np.clip(np.floor((dist + ahead_m) / self._spacing) + 1, 0, len(self._grid) - 1))
        if hi <= lo + 2:
            return None
        sl = slice(lo, hi)
        zeros = np.zeros(len(self._grid))
        return ReferenceWindow(
            offsets_m=self._grid[sl] - dist,
            throttle=np.asarray(self._ref.channels.get("Throttle", zeros)[sl], float),
            brake=np.asarray(self._ref.channels.get("Brake", zeros)[sl], float),
            speed=np.asarray(self._ref.channels.get("Speed", zeros)[sl], float),
        )
