"""Import Garage 61 CSV lap exports as first-class laps.

Garage 61 (garage61.net) exports a lap's telemetry as CSV with iRacing's
channel names — but without ``SessionTime``, ``LapDist`` or a lap counter
(it is a single lap sampled at 60 Hz, ``LapDistPct`` only, possibly wrapping
just past the finish line). This module reconstructs the missing axes:

* ``SessionTime`` — sample index / 60 Hz (validated against real exports);
* track length — integral of speed over the lap divided by the fraction of
  the lap covered, so ``LapDist = LapDistPct * length`` needs no external
  track database;
* exact S/F crossing anchors — extrapolated from the first/last samples,
  which makes the lap time and the delta engine work exactly like for laps
  parsed from ``.ibt`` files.

Driver/car/track/lap-time metadata comes from Garage 61's export filename
(``Garage 61 - <driver> - <car> - <track> - <MM.SS.mmm> - <lap id>.csv``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import LapData, SessionMeta
from .store import LapLibrary, LapRecord

__all__ = [
    "format_garage61_filename",
    "parse_garage61_filename",
    "read_garage61_csv",
    "import_garage61_csv",
]

log = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 60.0

_TIME_RE = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})$")
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$", re.IGNORECASE)

#: CSV columns that are integral/categorical
_INT_COLUMNS = {"Gear", "PositionType"}
#: Garage 61 name -> iRacing channel name (rest pass through unchanged)
_RENAMES = {"PositionType": "PlayerTrackSurface"}


def parse_garage61_filename(name: str) -> dict:
    """Extract metadata from a Garage 61 export filename; ``{}`` if it
    doesn't look like one."""
    stem = Path(name).stem
    parts = [p.strip() for p in stem.split(" - ")]
    if len(parts) < 6 or parts[0] != "Garage 61":
        return {}
    lap_id = parts[-1] if _ULID_RE.match(parts[-1]) else ""
    m = _TIME_RE.match(parts[-2])
    lap_time = int(m.group(1)) * 60.0 + float(f"{m.group(2)}.{m.group(3)}") if m else None
    return {
        "driver": parts[1],
        "car": parts[2],
        "track": " - ".join(parts[3:-2]),
        "lap_time": lap_time,
        "lap_id": lap_id,
    }


_ILLEGAL_FS_CHARS = str.maketrans({c: "_" for c in '/\\:*?"<>|'})


def _sanitize_component(text: str, strict: bool = True) -> str:
    """Make a name safe for the canonical filename.

    ``strict`` (driver/car): embedded ``" - "`` would shift the parser's
    field split, so it is replaced with an en-dash variant. The track field
    tolerates ``" - "`` (the parser joins the middle parts back).
    """
    clean = str(text).translate(_ILLEGAL_FS_CHARS).strip(" .")
    if strict:
        clean = clean.replace(" - ", " – ")
    return clean or "unknown"


def format_garage61_filename(
    driver: str, car: str, track: str, lap_time_s: float, lap_id: str
) -> str:
    """Compose the canonical Garage 61 export filename so that
    :func:`parse_garage61_filename` round-trips it exactly.

    Time is formatted millisecond-first to avoid the ``02.60.000`` rounding
    trap when seconds round up.
    """
    ms = round(float(lap_time_s) * 1000.0)
    minutes, rest = divmod(ms, 60_000)
    time_txt = f"{minutes:02d}.{rest // 1000:02d}.{rest % 1000:03d}"
    name = " - ".join(
        [
            "Garage 61",
            _sanitize_component(driver),
            _sanitize_component(car),
            _sanitize_component(track, strict=False),
            time_txt,
            str(lap_id),
        ]
    )
    return f"{name}.csv"


def _unwrap_pct(pct: np.ndarray) -> np.ndarray:
    """LapDistPct may wrap 0.999… -> 0.000… just past the finish line."""
    jumps = np.diff(pct) < -0.5
    return pct + np.concatenate([[0.0], np.cumsum(jumps)])


def read_garage61_csv(path: str | Path, tick_rate: float = SAMPLE_RATE_HZ) -> LapData:
    """Read one Garage 61 CSV lap export into a fully usable :class:`LapData`."""
    path = Path(path)
    frame = pd.read_csv(path)
    if "LapDistPct" not in frame.columns or "Speed" not in frame.columns:
        raise ValueError(f"{path.name} does not look like a Garage 61 lap export")
    n = len(frame)
    if n < 10:
        raise ValueError(f"{path.name}: too few samples ({n})")

    dt = 1.0 / tick_rate
    time = np.arange(n, dtype=np.float64) * dt
    pct = _unwrap_pct(frame["LapDistPct"].to_numpy(dtype=np.float64))
    pct = np.maximum.accumulate(pct)  # guard against single-sample jitter
    speed = frame["Speed"].to_numpy(dtype=np.float64)

    coverage = float(pct[-1] - pct[0])
    if coverage <= 0.5:
        raise ValueError(f"{path.name}: covers only {coverage:.1%} of a lap")
    track_length = float(np.sum(speed) * dt) / coverage
    dist = pct * track_length

    # exact S/F anchors: interpolate inside the data, extrapolate at the edges
    v0 = max(float(speed[0]), 1.0)
    t_start = float(time[0]) - (float(pct[0]) * track_length) / v0
    if pct[-1] >= 1.0:
        t_end = float(np.interp(1.0, pct, time))
    else:
        v1 = max(float(speed[-1]), 1.0)
        t_end = float(time[-1]) + ((1.0 - float(pct[-1])) * track_length) / v1
    reconstructed = t_end - t_start

    # The filename carries Garage 61's *official* lap time; prefer it — some
    # exports miss the last few % of telemetry samples, which would make the
    # constant-speed extrapolation above drift by seconds.
    info = parse_garage61_filename(path.name)
    official = info.get("lap_time")
    missing_m = max(0.0, 1.0 - coverage) * track_length
    if official and missing_m > 5.0:
        # physical consistency: the time left over after the recorded samples
        # must at least allow covering the missing distance at top speed
        t_remaining = official - n * dt
        t_min_needed = missing_m / max(float(speed.max()), 1.0)
        if t_remaining < t_min_needed:
            raise ValueError(
                f"{path.name}: telemetry does not match the labelled lap — the data "
                f"lasts {n * dt:.3f} s of the official {official:.3f} s but covers only "
                f"{coverage:.1%} of the track ({missing_m:.0f} m missing). Garage 61 "
                f"likely linked a different/misaligned telemetry stream to this lap; "
                f"re-export or pick another lap."
            )
    lap_time = official or reconstructed
    if info.get("lap_time") and abs(reconstructed - lap_time) > 0.1:
        log.warning(
            "%s: telemetry covers %.1f%% of the lap; using the official lap "
            "time %.3f s (naive reconstruction gave %.3f s)",
            path.name, coverage * 100.0, lap_time, reconstructed,
        )
    t_end = t_start + lap_time
    is_complete = coverage > 0.9  # officially timed; tail may be truncated

    channels: dict[str, np.ndarray] = {
        "SessionTime": time,
        "LapDist": dist.astype(np.float32),
        "LapDistPct": frame["LapDistPct"].to_numpy(dtype=np.float32),
        "Lap": np.ones(n, dtype=np.int32),
    }
    for column in frame.columns:
        if column == "LapDistPct":
            continue
        name = _RENAMES.get(column, column)
        values = frame[column]
        if values.dtype == bool:
            channels[name] = values.to_numpy()
        elif column in _INT_COLUMNS:
            channels[name] = values.to_numpy(dtype=np.int32)
        else:
            channels[name] = values.to_numpy(dtype=np.float32)

    meta = SessionMeta(
        path=path,
        track_name="",  # Garage 61 exports carry no iRacing-internal track id
        track_display_name=info.get("track", ""),
        track_length_m=track_length,
        car_screen_name=info.get("car", ""),
        driver_name=info.get("driver", ""),
        session_type="Garage 61 import",
        tick_rate=int(tick_rate),
        record_count=n,
    )
    return LapData(
        lap_number=1,
        start_idx=0,
        end_idx=n,
        lap_time=lap_time,
        crossing_start_time=t_start,
        crossing_end_time=t_end,
        is_complete=is_complete,
        channels=channels,
        meta=meta,
    )


def import_garage61_csv(
    library: LapLibrary,
    path: str | Path,
    tags: tuple[str, ...] = ("garage61",),
    is_reference: bool = False,
) -> LapRecord:
    """Read a Garage 61 CSV and store it in the reference library
    (idempotent per file)."""
    path = Path(path)
    lap = read_garage61_csv(path)
    return library.add_lap(lap, tags=tags, is_reference=is_reference, source_file=path)
