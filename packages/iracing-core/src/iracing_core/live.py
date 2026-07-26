"""Live telemetry ingest (shared by the overlay; harmless off-Windows).

A thin, typed wrapper around ``irsdk.IRSDK``. On a Windows machine with
iRacing running, :meth:`LiveTelemetry.connect` attaches to the sim's shared
memory; anywhere else it returns ``False`` instead of raising, so apps can
degrade gracefully. A ``test_file`` (memory dump or ``.ibt``) can be passed
for development/replay on any platform.

:class:`LiveLapRecorder` turns the stream of per-tick samples into complete
:class:`~iracing_core.models.LapData` objects, one per finished lap — the
building block for live lap-by-lap analysis (race-engineer updates).
"""

from __future__ import annotations

import logging
from typing import Any

import irsdk
import numpy as np

from .models import LapData, SessionMeta

__all__ = ["LiveTelemetry", "LiveLapRecorder", "LIVE_CHANNELS"]

log = logging.getLogger(__name__)

#: channels sampled by default for live lap analysis
LIVE_CHANNELS: tuple[str, ...] = (
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
)


class LiveTelemetry:
    """Connection to the running sim (or a replay file) with snapshot reads."""

    def __init__(self) -> None:
        self._ir = irsdk.IRSDK()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, test_file: str | None = None) -> bool:
        """Attach to iRacing shared memory (or a replay ``test_file``).

        Returns ``False`` — never raises — when the sim is not running or
        the platform has no iRacing shared memory (e.g. macOS/Linux dev).
        """
        try:
            self._connected = bool(self._ir.startup(test_file=test_file))
        except Exception as exc:
            log.debug("live telemetry unavailable: %s", exc)
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        try:
            self._ir.shutdown()
        except Exception:  # pragma: no cover - defensive
            pass
        self._connected = False

    def snapshot(self, names: list[str]) -> dict[str, Any]:
        """Read the latest value of each channel in ``names``."""
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        return {name: self._ir[name] for name in names}

    def session_info(self, key: str) -> Any:
        """Latest session-info section (e.g. ``'WeekendInfo'``)."""
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        return self._ir[key]

    def session_meta(self) -> SessionMeta:
        """Build a SessionMeta from the live session-info YAML."""
        info = {}
        for key in ("WeekendInfo", "DriverInfo", "SessionInfo", "SplitTimeInfo"):
            try:
                value = self.session_info(key)
            except Exception:
                value = None
            if value is not None:
                info[key] = value
        from pathlib import Path

        return SessionMeta.from_session_info(Path("<live>"), info)

    def snapshot_live_channels(self) -> dict[str, Any]:
        """Snapshot of the channels used for live lap analysis (missing
        channels are skipped)."""
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        snap = {}
        for name in LIVE_CHANNELS:
            try:
                value = self._ir[name]
            except Exception:
                value = None
            if value is not None:
                snap[name] = value
        return snap

    def start_disk_recording(self) -> bool:
        """Ask the sim to start writing ``.ibt`` telemetry (like Alt+L).

        Sends the ``TelemCommand`` broadcast message — Windows-only, since
        it goes through the Win32 message queue. Returns ``False`` (never
        raises) when not connected or off-Windows.
        """
        if not self._connected:
            return False
        try:
            self._ir.telem_command(irsdk.TelemCommandMode.start)
            return True
        except Exception as exc:
            log.debug("cannot send telemetry-start broadcast: %s", exc)
            return False


class LiveLapRecorder:
    """Assemble complete laps from a live stream of telemetry samples.

    Call :meth:`feed` with one sample dict per tick (any rate; lap timing is
    interpolated through the LapDistPct wrap, so even ~20 Hz polling stays
    accurate to a few hundredths). ``feed`` returns the *finished* lap
    whenever the ``Lap`` counter advances; :meth:`finish` flushes the final
    partial lap (e.g. the in-lap) at the end of a run.
    """

    _PIT_ZONE_PCT = 0.30

    def __init__(self, meta: SessionMeta | None = None):
        self.meta = meta
        self._rows: list[dict] = []
        self._lap_number: int | None = None
        self._t_start: float | None = None
        self._prev: dict | None = None

    def feed(self, sample: dict) -> LapData | None:
        lap_now = int(sample["Lap"])
        closed: LapData | None = None
        if self._lap_number is None:
            self._lap_number = lap_now
            if float(sample["LapDistPct"]) < 0.005:
                # the run starts exactly on the line
                self._t_start = float(sample["SessionTime"])
        elif lap_now != self._lap_number:
            t_cross = self._crossing_time(self._prev, sample)
            closed = self._close(t_end=t_cross)
            self._rows = []
            self._lap_number = lap_now
            self._t_start = t_cross
        self._rows.append(sample)
        self._prev = sample
        return closed

    def finish(self) -> LapData | None:
        """Flush the lap in progress (no end crossing — e.g. an in-lap)."""
        if not self._rows:
            return None
        lap = self._close(t_end=None)
        self._rows = []
        self._lap_number = None
        self._t_start = None
        return lap

    @staticmethod
    def _crossing_time(before: dict, after: dict) -> float:
        to_run = 1.0 - float(before["LapDistPct"])
        done = float(after["LapDistPct"])
        total = to_run + done
        frac = to_run / total if total > 1e-12 else 0.5
        t0, t1 = float(before["SessionTime"]), float(after["SessionTime"])
        return t0 + frac * (t1 - t0)

    def _close(self, t_end: float | None) -> LapData:
        rows = self._rows
        names = [name for name in rows[0] if all(name in r for r in rows)]
        channels = {name: np.asarray([r[name] for r in rows]) for name in names}
        n = len(rows)

        pct = channels["LapDistPct"].astype(float)
        lap_time = (t_end - self._t_start) if (t_end is not None and self._t_start is not None) else None

        on_pit = channels.get("OnPitRoad")
        touched = bool(on_pit.any()) if on_pit is not None else False
        pit_early = bool(on_pit[pct < self._PIT_ZONE_PCT].any()) if on_pit is not None else False
        pit_late = bool(on_pit[pct > 1.0 - self._PIT_ZONE_PCT].any()) if on_pit is not None else False
        covers = float(pct[0]) < 0.02 and float(pct[-1]) > 0.98

        return LapData(
            lap_number=int(self._lap_number or 0),
            start_idx=0,
            end_idx=n,
            lap_time=lap_time,
            crossing_start_time=self._t_start,
            crossing_end_time=t_end,
            is_out_lap=pit_early,
            is_in_lap=pit_late and t_end is None,
            is_complete=lap_time is not None and covers,
            touched_pits=touched,
            channels=channels,
            meta=self.meta,
        )
