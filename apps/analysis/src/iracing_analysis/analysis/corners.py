"""Corner detection and per-corner time-loss attribution.

Corners are found on the *reference* lap as local minima of smoothed speed,
validated by steering input, then widened into windows:

* window start — beginning of the braking zone (contiguous brake application
  leading to the apex), falling back to the start of the speed drop;
* window end — first sustained return to full throttle after the apex.

Overlapping windows are separated at the fastest point between two apexes.
The same windows are then used to measure per-lap metrics (braking point,
minimum/apex speed, throttle-up point, trail-brake overlap, steering) and to
integrate the delta-time gained/lost inside each corner, so corner deltas +
straight deltas telescope exactly to the total lap delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from iracing_core import AlignedLap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .workspace import Workspace

__all__ = [
    "CornerGeometry",
    "CornerLapMetrics",
    "CornerComparison",
    "SegmentDelta",
    "detect_corners",
    "corner_metrics",
    "segment_deltas",
    "analyze_corners",
    "rank_by_time_lost",
]

# tunables (all in SI; exposed here so the settings UI can override)
SMOOTH_SIGMA_M = 5.0
MIN_PROMINENCE_MS = 1.5  # m/s speed dip that counts as a corner
MIN_SEPARATION_M = 80.0
STEER_VALIDATION_RAD = 0.04
BRAKE_ON = 0.05
BRAKE_POINT_ON = 0.10
FULL_THROTTLE = 0.90
SUSTAIN_M = 10.0
BRAKE_SEARCH_EXTRA_M = 80.0


@dataclass(frozen=True)
class CornerGeometry:
    """A corner window on the common distance grid (from the reference lap)."""

    number: int
    start_m: float
    apex_m: float
    end_m: float
    direction: int  # +1 left, -1 right

    def label(self) -> str:
        return f"T{self.number}"


@dataclass(frozen=True)
class CornerLapMetrics:
    """How one lap drove one corner."""

    braking_point_m: float | None
    min_speed: float  # m/s
    min_speed_at_m: float
    throttle_full_m: float | None
    trail_brake_overlap_m: float
    mean_abs_steer: float


@dataclass(frozen=True)
class SegmentDelta:
    """Delta accumulated over one segment of the lap (corner or straight)."""

    label: str
    start_m: float
    end_m: float
    delta: float
    is_corner: bool


@dataclass(frozen=True)
class CornerComparison:
    """One corner: analysed lap vs reference, with the likely 'why'."""

    corner: CornerGeometry
    delta_total: float
    delta_entry: float  # window start → apex
    delta_exit: float  # apex → window end
    delta_carry: float  # window end → next corner: exit speed carried onto the straight
    metrics: CornerLapMetrics
    ref_metrics: CornerLapMetrics
    reasons: list[str]

    @property
    def time_lost(self) -> float:
        """Total attributable cost of the corner, including the run to the
        next corner (a slow exit keeps costing time down the whole straight)."""
        return self.delta_total + self.delta_carry


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def detect_corners(reference: AlignedLap) -> list[CornerGeometry]:
    grid = reference.grid
    spacing = float(grid[1] - grid[0])
    speed = reference.channels["Speed"].astype(float)
    steer = reference.channels.get("SteeringWheelAngle")
    brake = reference.channels.get("Brake")
    throttle = reference.channels.get("Throttle")

    sigma = max(1.0, SMOOTH_SIGMA_M / spacing)
    smooth = gaussian_filter1d(speed, sigma=sigma, mode="wrap")

    peaks, _ = find_peaks(
        -smooth,
        prominence=MIN_PROMINENCE_MS,
        distance=max(1, int(MIN_SEPARATION_M / spacing)),
    )
    if steer is not None:
        steer_smooth = gaussian_filter1d(np.abs(steer.astype(float)), sigma=sigma, mode="wrap")
        peaks = [p for p in peaks if steer_smooth[p] > STEER_VALIDATION_RAD]

    sustain = max(1, int(SUSTAIN_M / spacing))
    windows: list[tuple[int, int, int]] = []  # (start, apex, end) indices
    for apex in peaks:
        start = _braking_start(brake, smooth, apex, spacing)
        end = _throttle_full(throttle, apex, len(grid), sustain)
        windows.append((start, apex, end))

    # separate overlapping windows at the fastest point between the apexes
    for k in range(len(windows) - 1):
        s0, a0, e0 = windows[k]
        s1, a1, e1 = windows[k + 1]
        if e0 > s1:
            boundary = a0 + int(np.argmax(smooth[a0:a1])) if a1 > a0 else a0
            windows[k] = (s0, a0, min(e0, boundary))
            windows[k + 1] = (max(s1, boundary), a1, e1)

    corners = []
    for n, (start, apex, end) in enumerate(windows, start=1):
        direction = 1
        if steer is not None:
            direction = 1 if float(steer[apex]) >= 0 else -1
        corners.append(
            CornerGeometry(
                number=n,
                start_m=float(grid[start]),
                apex_m=float(grid[apex]),
                end_m=float(grid[min(end, len(grid) - 1)]),
                direction=direction,
            )
        )
    return corners


def _braking_start(
    brake: np.ndarray | None, smooth_speed: np.ndarray, apex: int, spacing: float
) -> int:
    """Index where the braking zone leading to ``apex`` begins."""
    gap = max(1, int(8.0 / spacing))
    lookback = int(500.0 / spacing)
    lo = max(0, apex - lookback)

    if brake is not None:
        on = brake[lo:apex] > BRAKE_ON
        if on.any():
            # walk back from the apex through the contiguous braking block
            idx = len(on) - 1
            seen_gap = 0
            start = None
            while idx >= 0:
                if on[idx]:
                    start = idx
                    seen_gap = 0
                else:
                    seen_gap += 1
                    if start is not None and seen_gap >= gap:
                        break
                idx -= 1
            if start is not None:
                return lo + start

    # no brake data / no braking: start where speed begins to fall
    idx = apex
    while idx > lo and smooth_speed[idx - 1] >= smooth_speed[idx]:
        idx -= 1
    return idx


def _throttle_full(throttle: np.ndarray | None, apex: int, n: int, sustain: int) -> int:
    """First index at/after ``apex`` with sustained full throttle."""
    if throttle is None:
        return min(apex + sustain, n - 1)
    full = throttle[apex:] > FULL_THROTTLE
    if len(full) >= sustain:
        window = np.convolve(full.astype(float), np.ones(sustain), mode="valid")
        hits = np.flatnonzero(window >= sustain)
        if len(hits):
            return min(apex + int(hits[0]), n - 1)
    return n - 1


# ---------------------------------------------------------------------------
# per-lap metrics
# ---------------------------------------------------------------------------


def corner_metrics(lap: AlignedLap, corner: CornerGeometry) -> CornerLapMetrics:
    grid = lap.grid
    spacing = float(grid[1] - grid[0])
    i0 = int(np.searchsorted(grid, corner.start_m))
    i1 = int(np.searchsorted(grid, corner.end_m))
    i1 = max(i1, i0 + 2)

    speed = lap.channels["Speed"].astype(float)
    brake = lap.channels.get("Brake")
    throttle = lap.channels.get("Throttle")
    steer = lap.channels.get("SteeringWheelAngle")

    # braking may start before the reference window: search a bit earlier
    b0 = max(0, i0 - int(BRAKE_SEARCH_EXTRA_M / spacing))
    braking_point = None
    if brake is not None:
        apex_i = int(np.searchsorted(grid, corner.apex_m))
        hits = np.flatnonzero(brake[b0:apex_i] > BRAKE_POINT_ON)
        if len(hits):
            braking_point = float(grid[b0 + hits[0]])

    seg = slice(i0, i1)
    min_i = i0 + int(np.argmin(speed[seg]))
    min_speed = float(speed[min_i])

    throttle_full = None
    if throttle is not None:
        sustain = max(1, int(SUSTAIN_M / spacing))
        j = _throttle_full(throttle, min_i, len(grid), sustain)
        end_limit = i1 + int(60.0 / spacing)
        if j < min(end_limit, len(grid) - 1):
            throttle_full = float(grid[j])

    trail = 0.0
    mean_steer = 0.0
    if steer is not None:
        abs_steer = np.abs(steer[seg].astype(float))
        mean_steer = float(abs_steer.mean())
        if brake is not None:
            steer_on = abs_steer > 0.3 * max(float(abs_steer.max()), 1e-9)
            trail = float(np.count_nonzero((brake[seg] > BRAKE_ON) & steer_on) * spacing)

    return CornerLapMetrics(
        braking_point_m=braking_point,
        min_speed=min_speed,
        min_speed_at_m=float(grid[min_i]),
        throttle_full_m=throttle_full,
        trail_brake_overlap_m=trail,
        mean_abs_steer=mean_steer,
    )


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------


def segment_deltas(
    workspace: Workspace, lap_index: int, corners: list[CornerGeometry]
) -> list[SegmentDelta]:
    """Partition the lap into corners and straights; the per-segment deltas
    telescope exactly to the total delta."""
    grid = workspace.grid
    delta = workspace.deltas[lap_index]

    edges: list[tuple[str, float, float, bool]] = []
    cursor = 0.0
    for corner in corners:
        if corner.start_m > cursor:
            edges.append((f"straight before {corner.label()}", cursor, corner.start_m, False))
        edges.append((corner.label(), corner.start_m, corner.end_m, True))
        cursor = corner.end_m
    end = float(grid[-1])
    if cursor < end:
        edges.append(("run to S/F", cursor, end, False))

    segments = []
    for label, start_m, end_m, is_corner in edges:
        i0 = int(np.searchsorted(grid, start_m))
        i1 = int(np.searchsorted(grid, end_m))
        i1 = min(i1, len(grid) - 1)
        segments.append(
            SegmentDelta(
                label=label,
                start_m=float(grid[i0]) if i0 < len(grid) else start_m,
                end_m=float(grid[i1]),
                delta=float(delta[i1] - delta[i0]),
                is_corner=is_corner,
            )
        )
    # snap the tiling to exact grid ends
    if segments:
        first = segments[0]
        segments[0] = SegmentDelta(first.label, 0.0, first.end_m, first.delta, first.is_corner)
    return segments


def _reasons(
    comp_delta: tuple[float, float, float, float],
    metrics: CornerLapMetrics,
    ref: CornerLapMetrics,
) -> list[str]:
    total, entry, exit_, carry = comp_delta
    reasons: list[str] = []

    if abs(total) > 0.02:
        if entry > 0.6 * abs(total) and entry > exit_:
            reasons.append("loses mainly on entry (braking phase → apex)")
        elif exit_ > 0.6 * abs(total) and exit_ > entry:
            reasons.append("loses mainly on exit (apex → full throttle)")
    if carry > 0.05:
        reasons.append(
            f"slow exit costs a further {carry:.2f} s down the following straight"
        )

    dv = ref.min_speed - metrics.min_speed
    if dv > 0.3:
        reasons.append(f"lower minimum/apex speed ({dv * 3.6:.1f} km/h slower)")
    elif dv < -0.3:
        reasons.append(f"higher minimum/apex speed ({-dv * 3.6:.1f} km/h faster)")

    if metrics.braking_point_m is not None and ref.braking_point_m is not None:
        db = metrics.braking_point_m - ref.braking_point_m
        if db < -5.0:
            reasons.append(f"brakes {-db:.0f} m earlier than the reference")
        elif db > 5.0:
            reasons.append(f"brakes {db:.0f} m later than the reference")

    if metrics.throttle_full_m is not None and ref.throttle_full_m is not None:
        dt_m = metrics.throttle_full_m - ref.throttle_full_m
        if dt_m > 8.0:
            reasons.append(f"back to full throttle {dt_m:.0f} m later (delayed application)")
        elif dt_m < -8.0:
            reasons.append(f"back to full throttle {-dt_m:.0f} m earlier")

    d_trail = metrics.trail_brake_overlap_m - ref.trail_brake_overlap_m
    if d_trail > 10.0:
        reasons.append(
            f"more trail-brake overlap ({metrics.trail_brake_overlap_m:.0f} m vs "
            f"{ref.trail_brake_overlap_m:.0f} m) — carrying brake deeper while turning"
        )

    if ref.mean_abs_steer > 1e-6:
        d_steer = (metrics.mean_abs_steer - ref.mean_abs_steer) / ref.mean_abs_steer
        if d_steer > 0.15:
            reasons.append("noticeably more steering input (tighter or longer line)")

    if not reasons:
        reasons.append("no single dominant cause — small differences throughout")
    return reasons


def analyze_corners(
    workspace: Workspace,
    lap_index: int,
    corners: list[CornerGeometry] | None = None,
) -> list[CornerComparison]:
    """Compare ``lap_index`` against the workspace reference, corner by corner."""
    if corners is None:
        corners = detect_corners(workspace.reference)
    lap = workspace.aligned.laps[lap_index]
    ref = workspace.reference
    grid = workspace.grid
    delta = workspace.deltas[lap_index]

    out: list[CornerComparison] = []
    for k, corner in enumerate(corners):
        i0 = int(np.searchsorted(grid, corner.start_m))
        ia = int(np.searchsorted(grid, corner.apex_m))
        i1 = min(int(np.searchsorted(grid, corner.end_m)), len(grid) - 1)
        next_start = corners[k + 1].start_m if k + 1 < len(corners) else float(grid[-1])
        ic = min(int(np.searchsorted(grid, next_start)), len(grid) - 1)
        d_entry = float(delta[ia] - delta[i0])
        d_exit = float(delta[i1] - delta[ia])
        d_carry = float(delta[ic] - delta[i1])
        metrics = corner_metrics(lap, corner)
        ref_metrics = corner_metrics(ref, corner)
        out.append(
            CornerComparison(
                corner=corner,
                delta_total=d_entry + d_exit,
                delta_entry=d_entry,
                delta_exit=d_exit,
                delta_carry=d_carry,
                metrics=metrics,
                ref_metrics=ref_metrics,
                reasons=_reasons(
                    (d_entry + d_exit, d_entry, d_exit, d_carry), metrics, ref_metrics
                ),
            )
        )
    return out


def rank_by_time_lost(comparisons: list[CornerComparison]) -> list[CornerComparison]:
    """Worst corner first, counting the loss carried onto the next straight."""
    return sorted(comparisons, key=lambda c: c.time_lost, reverse=True)
