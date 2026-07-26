"""Telemetry watcher: automatic collection of laps while you drive.

Runs as a small background agent (``iracing-agent``):

1. *Recording* (best effort, Windows + sim running): keeps iRacing's disk
   telemetry going by sending the ``TelemCommand start`` broadcast — the
   same as pressing Alt+L. For a zero-dependency alternative set
   ``irsdkEnableDisk=1`` in ``app.ini`` and this part becomes a no-op.
2. *Harvesting* (cross-platform): watches the telemetry folder; when a
   ``.ibt`` file stops growing for ``settle_seconds`` (session finished),
   its best/clean laps are imported into the reference library, tagged
   ``auto-import``. The library's ``(source_file, lap_number)`` uniqueness
   plus :meth:`LapLibrary.has_source` make the operation idempotent across
   agent restarts.

The engine is synchronous and clock-injectable so it can be driven
deterministically in tests; ``run_forever`` adds the polling loop.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .live import LiveTelemetry
from .sessions import default_telemetry_dir
from .store import LapLibrary, LapRecord

__all__ = ["WatcherConfig", "TelemetryWatcher", "main"]

log = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    telemetry_dir: Path
    library_dir: Path
    import_mode: str = "best"  # 'best' | 'clean' | 'all'
    tags: tuple[str, ...] = ("auto-import",)
    settle_seconds: float = 20.0  # file unchanged this long => session done
    poll_seconds: float = 5.0
    recording_check_seconds: float = 30.0
    ensure_recording: bool = True
    import_existing: bool = False  # also import files present before startup


@dataclass
class _FileState:
    size: int
    last_change: float
    processed: bool = False


class TelemetryWatcher:
    def __init__(
        self,
        config: WatcherConfig,
        library: LapLibrary | None = None,
        live: LiveTelemetry | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.library = library or LapLibrary(config.library_dir)
        self.live = live or LiveTelemetry()
        self._clock = clock
        self._files: dict[Path, _FileState] = {}
        self._first_scan_done = False
        self._next_recording_check = 0.0

    # -- harvesting ---------------------------------------------------------

    def scan_once(self) -> list[LapRecord]:
        """One pass over the telemetry folder; imports newly settled files."""
        now = self._clock()
        is_first_scan = not self._first_scan_done
        imported: list[LapRecord] = []

        for path in sorted(self.config.telemetry_dir.rglob("*.ibt")):
            try:
                size = path.stat().st_size
            except OSError:
                continue  # deleted between listing and stat

            state = self._files.get(path)
            if state is None:
                skip_backlog = is_first_scan and not self.config.import_existing
                state = _FileState(size=size, last_change=now, processed=skip_backlog)
                self._files[path] = state
                if skip_backlog:
                    log.debug("pre-existing file, not auto-importing: %s", path)
            elif size != state.size:
                state.size = size
                state.last_change = now  # still growing: session in progress

            if state.processed or now - state.last_change < self.config.settle_seconds:
                continue

            state.processed = True
            if self.library.has_source(path):
                log.debug("already in the library: %s", path)
                continue
            try:
                records = self.library.import_ibt(
                    path, laps=self.config.import_mode, tags=self.config.tags
                )
            except Exception as exc:
                log.warning("cannot import %s: %s", path, exc)
                continue
            if records:
                for rec in records:
                    log.info("imported %s", rec.label())
                imported.extend(records)
            else:
                log.info("no importable laps in %s", path.name)

        self._first_scan_done = True
        return imported

    # -- recording ------------------------------------------------------------

    def ensure_recording_once(self) -> bool:
        """Best effort: connect to the sim and (re)start disk telemetry."""
        if not self.config.ensure_recording:
            return False
        if not self.live.is_connected and not self.live.connect():
            return False
        ok = self.live.start_disk_recording()
        if not ok:
            self.live.disconnect()  # force a clean reconnect next time
        return ok

    # -- loop ---------------------------------------------------------------------

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        log.info(
            "watching %s (settle %.0fs, mode %s) -> library %s",
            self.config.telemetry_dir,
            self.config.settle_seconds,
            self.config.import_mode,
            self.config.library_dir,
        )
        while not stop.is_set():
            now = self._clock()
            if now >= self._next_recording_check:
                self._next_recording_check = now + self.config.recording_check_seconds
                self.ensure_recording_once()
            for record in self.scan_once():
                print(f"imported: {record.label()}")
            stop.wait(self.config.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="iracing-agent",
        description="Auto-record iRacing telemetry and auto-import finished "
        "sessions' laps into the reference library.",
    )
    parser.add_argument("--telemetry-dir", default=None, help="folder with .ibt files")
    parser.add_argument(
        "--library-dir",
        default=str(Path.home() / ".iracing_analysis" / "library"),
        help="reference library location (shared with the analysis app)",
    )
    parser.add_argument(
        "--mode", choices=("best", "clean", "all"), default="best",
        help="which laps to import per file (default: best)",
    )
    parser.add_argument("--tags", default="auto-import", help="comma-separated tags")
    parser.add_argument("--settle", type=float, default=20.0, help="seconds a file must stay unchanged")
    parser.add_argument("--poll", type=float, default=5.0, help="folder poll interval [s]")
    parser.add_argument("--once", action="store_true", help="single scan pass, then exit")
    parser.add_argument(
        "--import-existing", action="store_true",
        help="also import files that already exist at startup (backlog)",
    )
    parser.add_argument(
        "--no-recording", action="store_true",
        help="do not try to start the sim's disk telemetry (e.g. app.ini already does it)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = WatcherConfig(
        telemetry_dir=Path(args.telemetry_dir or default_telemetry_dir()),
        library_dir=Path(args.library_dir),
        import_mode=args.mode,
        tags=tuple(t.strip() for t in args.tags.split(",") if t.strip()),
        settle_seconds=args.settle,
        poll_seconds=args.poll,
        ensure_recording=not args.no_recording,
        import_existing=args.import_existing,
    )
    if not config.telemetry_dir.exists():
        print(f"error: telemetry folder {config.telemetry_dir} does not exist")
        return 1

    watcher = TelemetryWatcher(config)
    if args.once:
        records = watcher.scan_once()
        print(f"imported {len(records)} lap(s)")
        for rec in records:
            print(f"  {rec.label()}")
        return 0

    try:
        watcher.run_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
