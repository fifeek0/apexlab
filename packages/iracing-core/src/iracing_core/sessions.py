"""Telemetry-folder scanning, session grouping and lap extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ibt import IbtReader
from .models import LapData, SessionMeta

__all__ = [
    "DEFAULT_CHANNELS",
    "SessionGroup",
    "default_telemetry_dir",
    "scan_telemetry_dir",
    "extract_laps",
]

log = logging.getLogger(__name__)

#: channels loaded for analysis by default (when present in the file)
DEFAULT_CHANNELS: tuple[str, ...] = (
    "SessionTime",
    "Lap",
    "LapDist",
    "LapDistPct",
    "Speed",
    "Throttle",
    "Brake",
    "SteeringWheelAngle",
    "RPM",
    "Gear",
    "LatAccel",
    "LongAccel",
    "Lat",
    "Lon",
    "OnPitRoad",
    "PlayerTrackSurface",
    "FuelLevel",
    "LapCurrentLapTime",
    "VelocityX",
    "VelocityY",
    "Yaw",
    # tyres & conditions (present in .ibt, absent in Garage 61 CSV exports)
    "LFtempCM",
    "RFtempCM",
    "LRtempCM",
    "RRtempCM",
    "LFpressure",
    "RFpressure",
    "LRpressure",
    "RRpressure",
    "TrackTempCrew",
    "AirTemp",
)

#: fraction of the lap treated as "near the pits" for out/in-lap flags
_PIT_ZONE_PCT = 0.30


def default_telemetry_dir() -> Path:
    """iRacing's default telemetry folder (``Documents/iRacing/telemetry``).

    Works on Windows (where iRacing runs) and elsewhere (where imported
    ``.ibt`` files may be analysed).
    """
    return Path.home() / "Documents" / "iRacing" / "telemetry"


@dataclass
class SessionGroup:
    """All ``.ibt`` files recorded in the same iRacing session."""

    key: tuple
    files: list[SessionMeta] = field(default_factory=list)

    @property
    def meta(self) -> SessionMeta:
        return self.files[0]

    def label(self) -> str:
        m = self.meta
        when = m.session_start_date.strftime("%Y-%m-%d %H:%M") if m.session_start_date else "?"
        return f"{m.track_display_name} — {m.car_screen_name} — {m.session_type or 'Session'} — {when}"


def scan_telemetry_dir(root: str | Path) -> list[SessionGroup]:
    """Recursively scan ``root`` for ``.ibt`` files and group them by
    iRacing session (SessionID/SubSessionID, falling back to track+car)."""
    root = Path(root)
    groups: dict[tuple, SessionGroup] = {}
    if not root.exists():
        return []
    for path in sorted(root.rglob("*.ibt")):
        try:
            with IbtReader(path) as reader:
                meta = reader.meta
        except Exception as exc:  # unreadable/corrupt file: skip, don't crash the browser
            log.warning("skipping unreadable telemetry file %s: %s", path, exc)
            continue
        if meta.session_id is not None:
            key = ("sid", meta.session_id, meta.subsession_id)
        else:
            key = ("trackcar", meta.track_name, meta.car_path)
        groups.setdefault(key, SessionGroup(key=key)).files.append(meta)
    return sorted(
        groups.values(),
        key=lambda g: (g.meta.session_start_date is None, g.meta.session_start_date, str(g.meta.path)),
    )


def _crossing_time(time: np.ndarray, pct: np.ndarray, boundary: int) -> float:
    """Exact S/F crossing time at a lap boundary index (sub-sample precision).

    ``boundary`` is the index of the first sample of the new lap; the line
    was crossed between ``boundary - 1`` and ``boundary``. Linear
    interpolation of LapDistPct through the wrap gives the crossing instant.
    """
    i, j = boundary - 1, boundary
    before = 1.0 - float(pct[i])  # distance fraction still to run at sample i
    after = float(pct[j])  # fraction already run at sample j
    total = before + after
    frac = before / total if total > 1e-12 else 0.5
    return float(time[i]) + frac * float(time[j] - time[i])


def extract_laps(
    reader: IbtReader,
    extra_channels: tuple[str, ...] = (),
) -> list[LapData]:
    """Split a telemetry file into laps with exact lap times and flags.

    Laps are cut at ``Lap`` counter changes; lap times are refined to
    sub-sample precision by interpolating the LapDistPct wrap, so a clean
    lap's time matches iRacing's to within a few milliseconds.
    """
    wanted = [
        name
        for name in (*DEFAULT_CHANNELS, *extra_channels)
        if reader.has_channel(name)
    ]
    channels = reader.get_channels(wanted)

    lap_ch = channels["Lap"]
    time = channels["SessionTime"]
    pct = channels["LapDistPct"]
    on_pit = channels.get("OnPitRoad")
    n = len(lap_ch)
    if n == 0:
        return []

    boundaries = (np.flatnonzero(np.diff(lap_ch) != 0) + 1).tolist()
    starts = [0, *boundaries]
    ends = [*boundaries, n]

    meta = reader.meta
    laps: list[LapData] = []
    for k, (start, end) in enumerate(zip(starts, ends)):
        has_start_crossing = start != 0
        has_end_crossing = end != n

        t_start: float | None = None
        t_end: float | None = None
        if has_start_crossing:
            t_start = _crossing_time(time, pct, start)
        elif pct[start] < 0.005:
            # the file starts exactly on the line (e.g. synthetic sessions)
            t_start = float(time[start])
        if has_end_crossing:
            t_end = _crossing_time(time, pct, end)
        lap_time = (t_end - t_start) if t_start is not None and t_end is not None else None

        seg_pct = pct[start:end]
        pit_seg = on_pit[start:end] if on_pit is not None else None
        touched_pits = bool(pit_seg.any()) if pit_seg is not None else False
        pit_early = bool(pit_seg[seg_pct < _PIT_ZONE_PCT].any()) if pit_seg is not None else False
        pit_late = bool(pit_seg[seg_pct > 1.0 - _PIT_ZONE_PCT].any()) if pit_seg is not None else False

        covers_lap = float(seg_pct[0]) < 0.02 and float(seg_pct[-1]) > 0.98
        is_complete = lap_time is not None and covers_lap

        laps.append(
            LapData(
                lap_number=int(lap_ch[start]),
                start_idx=start,
                end_idx=end,
                lap_time=lap_time,
                crossing_start_time=t_start,
                crossing_end_time=t_end,
                is_out_lap=pit_early,
                is_in_lap=pit_late and not has_end_crossing,
                is_complete=is_complete,
                touched_pits=touched_pits,
                channels={name: arr[start:end] for name, arr in channels.items()},
                meta=meta,
            )
        )
    return laps
