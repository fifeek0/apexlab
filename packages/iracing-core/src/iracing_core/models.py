"""Shared data model: session metadata and per-lap telemetry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["SessionMeta", "LapData", "parse_track_length_m"]

_TRACK_LEN_RE = re.compile(r"([\d.]+)\s*(km|mi)", re.IGNORECASE)


def parse_track_length_m(text: str | None) -> float | None:
    """Parse iRacing's ``WeekendInfo.TrackLength`` string ('3.21 km') to metres."""
    if not text:
        return None
    m = _TRACK_LEN_RE.search(str(text))
    if not m:
        return None
    value = float(m.group(1))
    return value * (1000.0 if m.group(2).lower() == "km" else 1609.344)


@dataclass
class SessionMeta:
    """Metadata of one ``.ibt`` file, extracted from the embedded YAML."""

    path: Path
    track_name: str = ""
    track_display_name: str = ""
    track_config: str = ""
    track_length_m: float | None = None
    car_path: str = ""
    car_screen_name: str = ""
    driver_name: str = ""
    session_type: str = ""
    session_id: int | None = None
    subsession_id: int | None = None
    session_start_date: datetime | None = None
    tick_rate: int = 60
    record_count: int = 0
    sector_starts_pct: list[float] = field(default_factory=list)
    session_info: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_session_info(
        cls,
        path: Path,
        info: dict,
        *,
        tick_rate: int = 60,
        record_count: int = 0,
        session_start_unix: int | None = None,
    ) -> "SessionMeta":
        weekend = info.get("WeekendInfo") or {}
        driver_info = info.get("DriverInfo") or {}
        sessions = (info.get("SessionInfo") or {}).get("Sessions") or []
        sectors = (info.get("SplitTimeInfo") or {}).get("Sectors") or []

        driver_name = ""
        car_screen_name = ""
        car_path = ""
        drivers = driver_info.get("Drivers") or []
        car_idx = driver_info.get("DriverCarIdx")
        me = next((d for d in drivers if d.get("CarIdx") == car_idx), None)
        if me:
            driver_name = str(me.get("UserName", ""))
            car_screen_name = str(me.get("CarScreenName", ""))
            car_path = str(me.get("CarPath", ""))

        started = None
        if session_start_unix:
            started = datetime.fromtimestamp(session_start_unix, tz=timezone.utc)

        return cls(
            path=path,
            track_name=str(weekend.get("TrackName", "")),
            track_display_name=str(weekend.get("TrackDisplayName", "")),
            track_config=str(weekend.get("TrackConfigName", "") or ""),
            track_length_m=parse_track_length_m(weekend.get("TrackLength")),
            car_path=car_path,
            car_screen_name=car_screen_name,
            driver_name=driver_name,
            session_type=str(sessions[0].get("SessionType", "")) if sessions else "",
            session_id=weekend.get("SessionID"),
            subsession_id=weekend.get("SubSessionID"),
            session_start_date=started,
            tick_rate=tick_rate,
            record_count=record_count,
            sector_starts_pct=[float(s["SectorStartPct"]) for s in sectors if "SectorStartPct" in s],
            session_info=info,
        )


@dataclass
class LapData:
    """One lap sliced out of a telemetry file.

    ``channels`` values are numpy views into the session-level arrays, so
    holding many laps costs no extra memory.
    """

    lap_number: int
    start_idx: int
    end_idx: int  # exclusive
    lap_time: float | None = None  # exact S/F-to-S/F time; None if incomplete
    crossing_start_time: float | None = None  # SessionTime of the starting S/F crossing
    crossing_end_time: float | None = None  # SessionTime of the ending S/F crossing
    is_out_lap: bool = False
    is_in_lap: bool = False
    is_complete: bool = False
    touched_pits: bool = False
    channels: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    meta: SessionMeta | None = field(default=None, repr=False)

    @property
    def is_clean(self) -> bool:
        """A complete flying lap with no pit-lane contact."""
        return self.is_complete and not self.touched_pits

    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx

    def channel(self, name: str) -> np.ndarray:
        try:
            return self.channels[name]
        except KeyError:
            raise KeyError(
                f"channel {name!r} not loaded for lap {self.lap_number}; "
                f"available: {sorted(self.channels)}"
            ) from None

    def label(self) -> str:
        """Short human-readable label, e.g. 'L3 1:32.456'."""
        if self.lap_time is None:
            t = "--:--.---"
        else:
            mins, secs = divmod(self.lap_time, 60.0)
            t = f"{int(mins)}:{secs:06.3f}"
        return f"L{self.lap_number} {t}"
