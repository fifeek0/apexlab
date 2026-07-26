"""iracing_core — shared telemetry core for the overlay and analysis apps.

Provides live telemetry ingest, ``.ibt`` file import, the lap store /
reference-lap library and distance-based lap alignment + delta math.
Both the real-time overlay and the post-session analysis app depend on
this package; neither duplicates its code.
"""

from .alignment import AlignedLap, AlignedLapSet, align_lap, align_laps, delta_time
from .garage61 import import_garage61_csv, read_garage61_csv
from .ibt import IbtReader, read_session_info
from .live import LIVE_CHANNELS, LiveLapRecorder, LiveTelemetry
from .models import LapData, SessionMeta, parse_track_length_m
from .sessions import (
    DEFAULT_CHANNELS,
    SessionGroup,
    default_telemetry_dir,
    extract_laps,
    scan_telemetry_dir,
)
from .store import LapLibrary, LapRecord
from .watcher import TelemetryWatcher, WatcherConfig

__version__ = "0.1.0"

__all__ = [
    "AlignedLap",
    "AlignedLapSet",
    "DEFAULT_CHANNELS",
    "IbtReader",
    "LIVE_CHANNELS",
    "LapLibrary",
    "LapRecord",
    "align_lap",
    "align_laps",
    "delta_time",
    "LapData",
    "LiveLapRecorder",
    "LiveTelemetry",
    "SessionGroup",
    "SessionMeta",
    "TelemetryWatcher",
    "WatcherConfig",
    "default_telemetry_dir",
    "import_garage61_csv",
    "read_garage61_csv",
    "extract_laps",
    "parse_track_length_m",
    "read_session_info",
    "scan_telemetry_dir",
    "__version__",
]
