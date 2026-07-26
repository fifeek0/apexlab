"""Synthetic iRacing telemetry generator.

Builds a geometrically closed fantasy circuit (a polygon whose vertices are
rounded into constant-radius arcs — closure is guaranteed by construction and
the corner count is exact) and simulates physically plausible laps over it
with a classic three-pass speed profile:

1. grip limit      v_lim(s) = sqrt(a_lat_max / |kappa(s)|)
2. braking pass    backward integration limited by a_brake
3. traction pass   forward integration limited by drag-reduced engine accel

The generated channels use iRacing's names, units and sign conventions so
they can be written to a real ``.ibt`` file (see
:mod:`iracing_core.testing.ibt_writer`) and parsed back with pyirsdk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "Corner",
    "SyntheticTrack",
    "DriverParams",
    "SyntheticSession",
    "default_track",
    "build_session",
]

SAMPLE_RATE_HZ = 60
_PROFILE_DS = 1.0  # m, resolution of the speed-profile grid

# Reference geodetic origin for the fantasy circuit.
_LAT0_DEG = 50.0
_LON0_DEG = 19.5
_M_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class Corner:
    """One rounded polygon vertex = one real corner of the circuit."""

    index: int
    apex_s: float  # distance of the arc midpoint from S/F, m
    start_s: float  # arc entry, m
    end_s: float  # arc exit, m
    radius: float  # m
    direction: int  # +1 left, -1 right (iRacing steering sign convention)

    @property
    def arc_length(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class _Segment:
    start_s: float
    length: float
    x0: float
    y0: float
    heading0: float  # rad, CCW from +x
    curvature: float  # 1/m, signed (+ left); 0 for straights


class SyntheticTrack:
    """Closed circuit built from a simple polygon with rounded corners.

    ``radii[i]`` rounds vertex ``i``. Straights connect the arc tangent
    points, so the loop closes exactly and ``n_corners == len(vertices)``.
    """

    def __init__(self, vertices: list[tuple[float, float]], radii: list[float]) -> None:
        if len(vertices) != len(radii):
            raise ValueError("one radius per polygon vertex required")
        pts = [np.asarray(p, dtype=float) for p in vertices]
        k = len(pts)

        turn_sum = 0.0
        tangent_pts: list[tuple[np.ndarray, np.ndarray, float, int]] = []
        for i in range(k):
            prev_pt, pt, next_pt = pts[i - 1], pts[i], pts[(i + 1) % k]
            u = pt - prev_pt
            w = next_pt - pt
            u /= np.linalg.norm(u)
            w /= np.linalg.norm(w)
            delta = math.atan2(u[0] * w[1] - u[1] * w[0], float(np.dot(u, w)))
            if abs(delta) < 1e-6:
                raise ValueError(f"vertex {i} is collinear; remove it")
            turn_sum += delta
            t = radii[i] * math.tan(abs(delta) / 2.0)
            t_in = pt - u * t  # tangent point on the incoming edge
            t_out = pt + w * t  # tangent point on the outgoing edge
            tangent_pts.append((t_in, t_out, delta, i))
        if abs(abs(turn_sum) - 2.0 * math.pi) > 1e-6:
            raise ValueError("polygon does not wind exactly once; check the vertex list")

        # Walk the loop: straight into vertex i, then the arc around vertex i.
        segments: list[_Segment] = []
        corner_raw: list[tuple[int, float, float, float, int]] = []
        s = 0.0
        pos = tangent_pts[-1][1].copy()  # start at the exit of the last corner
        for i in range(k):
            t_in, t_out, delta, _ = tangent_pts[i]
            chord = t_in - pos
            straight_len = float(np.linalg.norm(chord))
            heading = math.atan2(chord[1], chord[0])
            if straight_len < 1.0:
                raise ValueError(
                    f"edge into vertex {i} too short for radius {radii[i]} m"
                )
            segments.append(_Segment(s, straight_len, pos[0], pos[1], heading, 0.0))
            s += straight_len

            arc_len = radii[i] * abs(delta)
            curvature = math.copysign(1.0 / radii[i], delta)
            segments.append(_Segment(s, arc_len, t_in[0], t_in[1], heading, curvature))
            corner_raw.append((i, s, s + arc_len, radii[i], 1 if delta > 0 else -1))
            s += arc_len
            pos = t_out.copy()

        self._segments = segments
        self._raw_length = s
        # Put S/F in the middle of the longest straight, like a real circuit.
        longest = max((seg for seg in segments if seg.curvature == 0.0), key=lambda g: g.length)
        self._s0 = (longest.start_s + longest.length / 2.0) % s

        self.corners: list[Corner] = sorted(
            (
                Corner(
                    index=idx,
                    apex_s=(start + (end - start) / 2.0 - self._s0) % s,
                    start_s=(start - self._s0) % s,
                    end_s=(end - self._s0) % s,
                    radius=r,
                    direction=direction,
                )
                for idx, start, end, r, direction in corner_raw
            ),
            key=lambda c: c.apex_s,
        )
        self.corners = [replace(c, index=i + 1) for i, c in enumerate(self.corners)]

        self._seg_starts = np.array([seg.start_s for seg in segments])

    @property
    def length(self) -> float:
        return self._raw_length

    @property
    def n_corners(self) -> int:
        return len(self.corners)

    def _segment_index(self, s_raw: np.ndarray) -> np.ndarray:
        return np.clip(
            np.searchsorted(self._seg_starts, s_raw, side="right") - 1,
            0,
            len(self._segments) - 1,
        )

    def curvature(self, s: float | np.ndarray) -> np.ndarray:
        """Signed curvature (1/m, + for left) at distance ``s`` from S/F."""
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        s_raw = (s_arr + self._s0) % self._raw_length
        idx = self._segment_index(s_raw)
        kappa = np.array([self._segments[i].curvature for i in idx])
        return kappa if np.ndim(s) else kappa

    def xy(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Cartesian position (m) at distance ``s`` from S/F."""
        scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        s_raw = (s_arr + self._s0) % self._raw_length
        idx = self._segment_index(s_raw)
        x = np.empty_like(s_arr)
        y = np.empty_like(s_arr)
        for i, seg in enumerate(self._segments):
            mask = idx == i
            if not mask.any():
                continue
            ds = s_raw[mask] - seg.start_s
            if seg.curvature == 0.0:
                x[mask] = seg.x0 + ds * math.cos(seg.heading0)
                y[mask] = seg.y0 + ds * math.sin(seg.heading0)
            else:
                r = 1.0 / seg.curvature  # signed radius
                # centre is 90 deg to the left (signed) of the heading
                cx = seg.x0 - r * math.sin(seg.heading0)
                cy = seg.y0 + r * math.cos(seg.heading0)
                phi0 = math.atan2(seg.y0 - cy, seg.x0 - cx)
                phi = phi0 + ds * seg.curvature
                x[mask] = cx + abs(r) * np.cos(phi)
                y[mask] = cy + abs(r) * np.sin(phi)
        if scalar:
            return float(x[0]), float(y[0])
        return x, y

    def latlon(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.xy(s)
        lat = _LAT0_DEG + y / _M_PER_DEG_LAT
        lon = _LON0_DEG + x / (_M_PER_DEG_LAT * math.cos(math.radians(_LAT0_DEG)))
        return lat, lon


def default_track() -> SyntheticTrack:
    """~3.2 km, 10-corner fantasy circuit ('Fantasia International').

    Every radius is small enough that each corner forces a lift, so speed-
    minimum-based corner detection must find exactly ``n_corners`` corners.
    """
    vertices = [
        (0.0, 0.0),
        (620.0, 0.0),
        (760.0, 180.0),
        (700.0, 420.0),
        (420.0, 520.0),
        (520.0, 760.0),
        (150.0, 900.0),
        (-140.0, 700.0),
        (-70.0, 320.0),
        (-230.0, 60.0),
    ]
    radii = [45.0, 30.0, 22.0, 35.0, 28.0, 50.0, 40.0, 26.0, 33.0, 24.0]
    return SyntheticTrack(vertices, radii)


@dataclass(frozen=True)
class DriverParams:
    """Simple point-mass car/driver model."""

    a_lat_max: float = 20.0  # m/s^2 lateral grip
    a_brake: float = 22.0  # m/s^2 braking decel
    a_engine: float = 9.0  # m/s^2 tractive accel at v=0
    v_terminal: float = 78.0  # m/s drag-limited top speed
    steer_ratio: float = 12.0
    wheelbase: float = 2.6  # m

    def scaled(self, rng: np.random.Generator, spread: float = 0.012) -> "DriverParams":
        """A per-lap variation of this driver (slightly different limits)."""
        f = lambda: float(1.0 + rng.uniform(-spread, spread))  # noqa: E731
        return replace(
            self,
            a_lat_max=self.a_lat_max * f(),
            a_brake=self.a_brake * f(),
            a_engine=self.a_engine * f(),
        )


def _speed_profile(track: SyntheticTrack, params: DriverParams, rng: np.random.Generator | None) -> np.ndarray:
    """Quasi-steady-state speed profile v(s) on a 1 m grid (periodic)."""
    n = int(math.ceil(track.length / _PROFILE_DS))
    s = np.arange(n) * _PROFILE_DS
    kappa = np.abs(track.curvature(s))
    with np.errstate(divide="ignore"):
        v_lim = np.where(kappa > 1e-9, np.sqrt(params.a_lat_max / np.maximum(kappa, 1e-9)), np.inf)
    v_lim = np.minimum(v_lim, params.v_terminal)

    if rng is not None:
        # smooth per-lap 'line quality' field, ~ +/-0.7 %
        raw = rng.standard_normal(12)
        phases = rng.uniform(0, 2 * math.pi, 12)
        wave = sum(
            raw[i] * np.sin(2 * math.pi * (i + 1) * s / track.length + phases[i])
            for i in range(12)
        )
        v_lim = v_lim * (1.0 + 0.002 * wave)

    v = v_lim.copy()
    # braking (backward, run twice around the loop for periodicity)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            v_next = v[(i + 1) % n]
            v[i] = min(v[i], math.sqrt(v_next**2 + 2.0 * params.a_brake * _PROFILE_DS))
    # traction (forward, drag-limited engine)
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            a = params.a_engine * max(0.05, 1.0 - (v[i] / params.v_terminal) ** 2)
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * a * _PROFILE_DS))
    return v


@dataclass
class SyntheticSession:
    """Ground-truth product of :func:`build_session`."""

    channels: dict[str, np.ndarray]
    lap_times: list[float]  # flying laps only, exact crossing-to-crossing times
    session_info: dict
    track: SyntheticTrack
    driver_params: list[DriverParams] = field(default_factory=list)


def _default_session_info(
    track: SyntheticTrack,
    *,
    track_name: str,
    track_display_name: str,
    session_id: int,
    driver_name: str,
    car_screen_name: str,
    car_path: str,
) -> dict:
    return {
        "WeekendInfo": {
            "TrackName": track_name,
            "TrackID": 999,
            "TrackDisplayName": track_display_name,
            "TrackDisplayShortName": track_display_name.split()[0],
            "TrackConfigName": "Grand Prix",
            "TrackLength": f"{track.length / 1000.0:.2f} km",
            "TrackNumTurns": track.n_corners,
            "TrackAltitude": "210.00 m",
            "TrackAirTemp": "22.00 C",
            "TrackSurfaceTemp": "31.00 C",
            "SessionID": session_id,
            "SubSessionID": session_id * 10 + 1,
            "SeasonID": 5000,
            "SeriesID": 0,
            "Official": 0,
            "EventType": "Test",
        },
        "SessionInfo": {
            "Sessions": [
                {
                    "SessionNum": 0,
                    "SessionType": "Practice",
                    "SessionName": "TESTING",
                    "SessionLaps": "unlimited",
                }
            ]
        },
        "DriverInfo": {
            "DriverCarIdx": 0,
            "DriverUserID": 123456,
            "Drivers": [
                {
                    "CarIdx": 0,
                    "UserName": driver_name,
                    "UserID": 123456,
                    "CarScreenName": car_screen_name,
                    "CarPath": car_path,
                    "IRating": 3500,
                    "LicString": "A 4.99",
                }
            ],
        },
        "SplitTimeInfo": {
            "Sectors": [
                {"SectorNum": 0, "SectorStartPct": 0.0},
                {"SectorNum": 1, "SectorStartPct": 0.31},
                {"SectorNum": 2, "SectorStartPct": 0.66},
            ]
        },
    }


def build_session(
    track: SyntheticTrack,
    n_laps: int = 3,
    seed: int = 42,
    with_out_in_laps: bool = True,
    base_params: DriverParams | None = None,
    param_spread: float = 0.012,
    per_lap_params: list[DriverParams] | None = None,
    track_name: str = "fantasia full",
    track_display_name: str = "Fantasia International",
    session_id: int = 4242,
    driver_name: str = "Test Driver",
    car_screen_name: str = "Formula Fable GT3",
    car_path: str = "formulafable",
) -> SyntheticSession:
    """Simulate a practice run: optional out-lap, ``n_laps`` flying laps, optional in-lap."""
    rng = np.random.default_rng(seed)
    base = base_params or DriverParams()
    track_len = track.length
    dt = 1.0 / SAMPLE_RATE_HZ
    pit_zone = 350.0  # m of pit lane influence at lap start/end
    v_pit = 23.0  # m/s pit-lane speed

    n_flying = n_laps
    lap_specs: list[tuple[str, DriverParams]] = []
    if per_lap_params is not None and len(per_lap_params) != n_flying:
        raise ValueError("per_lap_params must have one entry per flying lap")

    def _flying_params(i: int) -> DriverParams:
        if per_lap_params is not None:
            return per_lap_params[i]
        return base.scaled(rng, param_spread)

    if with_out_in_laps:
        lap_specs.append(("out", base.scaled(rng, param_spread)))
    lap_specs += [("flying", _flying_params(i)) for i in range(n_flying)]
    if with_out_in_laps:
        lap_specs.append(("in", base.scaled(rng, param_spread)))

    profiles = [_speed_profile(track, p, rng) for _, p in lap_specs]
    prof_s = np.arange(len(profiles[0])) * _PROFILE_DS

    def lap_speed(lap_idx: int, s: float) -> float:
        v = float(np.interp(s % track_len, prof_s, profiles[lap_idx], period=track_len))
        # blend from the previous lap's profile over the first 150 m
        if lap_idx > 0 and s < 150.0:
            v_prev = float(
                np.interp(s % track_len, prof_s, profiles[lap_idx - 1], period=track_len)
            )
            w = s / 150.0
            v = (1.0 - w) * v_prev + w * v
        kind = lap_specs[lap_idx][0]
        if kind == "out":
            if s < pit_zone:
                v = min(v, v_pit)
            else:
                v = min(v, v_pit + (s - pit_zone) * 0.12)
        elif kind == "in":
            dist_to_stall = (track_len - 120.0) - s
            if dist_to_stall < pit_zone:
                v = min(v, max(3.0, v_pit * min(1.0, dist_to_stall / 60.0 + 0.15)))
        return max(v, 1.0)

    # --- simulate ------------------------------------------------------
    t = 0.0
    s = 0.0
    lap_idx = 0
    lap_counter = 0 if with_out_in_laps else 1
    crossings: list[float] = []

    rows_t: list[float] = []
    rows_s: list[float] = []
    rows_v: list[float] = []
    rows_lap: list[int] = []
    rows_lapidx: list[int] = []

    while lap_idx < len(lap_specs):
        kind = lap_specs[lap_idx][0]
        v = lap_speed(lap_idx, s)
        rows_t.append(t)
        rows_s.append(s)
        rows_v.append(v)
        rows_lap.append(lap_counter)
        rows_lapidx.append(lap_idx)

        if kind == "in" and s >= track_len - 125.0:
            break  # parked in the pit stall

        s_next = s + v * dt
        t += dt
        if s_next >= track_len:
            crossings.append(t - dt + (track_len - s) / v)
            s_next -= track_len
            lap_idx += 1
            lap_counter += 1
        s = s_next

    if lap_idx >= len(lap_specs):
        # the run ended by crossing the line: record one post-crossing sample
        # so the final Lap increment is visible in the channel data
        rows_t.append(t)
        rows_s.append(s)
        rows_v.append(rows_v[-1])
        rows_lap.append(lap_counter)
        rows_lapidx.append(len(lap_specs) - 1)

    time = np.asarray(rows_t)
    dist = np.asarray(rows_s)
    speed = np.asarray(rows_v)
    lap_arr = np.asarray(rows_lap, dtype=np.int32)
    lapidx = np.asarray(rows_lapidx)

    # --- derived channels ---------------------------------------------
    a_long = np.gradient(speed, dt)
    kappa = track.curvature(dist)
    a_lat = speed**2 * kappa

    drag = base.a_engine * (speed / base.v_terminal) ** 2
    a_wheel = a_long + drag
    throttle = np.clip(a_wheel / base.a_engine, 0.0, 1.0)
    brake = np.clip(-a_long / base.a_brake, 0.0, 1.0)
    coasting = a_wheel >= 0.0
    brake[coasting] = 0.0
    throttle[~coasting] = 0.0

    steer = base.steer_ratio * np.arctan(base.wheelbase * kappa)

    gear_edges = np.array([0.0, 15.0, 22.0, 30.0, 39.0, 50.0, 62.0])
    gear = np.searchsorted(gear_edges, speed, side="right").astype(np.int32)
    gear = np.clip(gear, 1, 7)
    hi = np.append(gear_edges[1:], base.v_terminal + 5.0)[gear - 1]
    rpm = np.maximum(1200.0, 7200.0 * speed / hi)

    lat_deg, lon_deg = track.latlon(dist)

    on_pit = np.zeros(len(time), dtype=bool)
    if with_out_in_laps:
        on_pit |= (lapidx == 0) & (dist < pit_zone)
        on_pit |= (lapidx == len(lap_specs) - 1) & (
            dist > track_len - 120.0 - pit_zone
        )
    surface = np.where(on_pit, 2, 3).astype(np.int32)

    fuel = 45.0 - 2.8 * (lap_arr - lap_arr[0] + dist / track_len)

    channels: dict[str, np.ndarray] = {
        "SessionTime": time.astype(np.float64),
        "SessionTick": (np.arange(len(time)) + 1).astype(np.int32),
        "Lap": lap_arr,
        "LapDist": dist.astype(np.float32),
        "LapDistPct": (dist / track_len).astype(np.float32),
        "Speed": speed.astype(np.float32),
        "Throttle": throttle.astype(np.float32),
        "Brake": brake.astype(np.float32),
        "SteeringWheelAngle": steer.astype(np.float32),
        "RPM": rpm.astype(np.float32),
        "Gear": gear,
        "LatAccel": a_lat.astype(np.float32),
        "LongAccel": a_long.astype(np.float32),
        "Lat": lat_deg.astype(np.float64),
        "Lon": lon_deg.astype(np.float64),
        "OnPitRoad": on_pit,
        "PlayerTrackSurface": surface,
        "FuelLevel": fuel.astype(np.float32),
    }

    if not with_out_in_laps:
        # the run starts exactly on the line, so t=0 is the first "crossing"
        crossings = [0.0, *crossings]
    lap_times = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    flying_times = lap_times[:n_flying]

    info = _default_session_info(
        track,
        track_name=track_name,
        track_display_name=track_display_name,
        session_id=session_id,
        driver_name=driver_name,
        car_screen_name=car_screen_name,
        car_path=car_path,
    )
    return SyntheticSession(
        channels=channels,
        lap_times=flying_times,
        session_info=info,
        track=track,
        driver_params=[p for _, p in lap_specs],
    )
