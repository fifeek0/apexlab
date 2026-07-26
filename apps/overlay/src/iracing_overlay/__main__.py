"""iracing-overlay: Bloops-style live comparison overlay.

Modes:

* (default) live — waits for iRacing, resolves the best matching reference
  lap from the library and shows the overlay (delta bar, input trace with
  reference preview, gear hint, braking-zone audio cues);
* ``--replay stint.ibt`` — drives the same overlay from a recorded stint at
  real-time speed (any OS; perfect for trying it without the sim);
* ``--check`` / ``--watch`` — quick shared-core diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import time

import iracing_core
from iracing_core import IbtReader, LapData, LapLibrary, LiveTelemetry
from iracing_core.live import LIVE_CHANNELS
from iracing_core.live_compare import LiveComparison

log = logging.getLogger(__name__)


def _resolve_reference(library: LapLibrary, args, meta=None) -> LapData:
    if args.reference_id is not None:
        record = library.get_record(args.reference_id)
        print(f"reference: {record.label()}")
        return library.get_lap(args.reference_id)
    candidates = []
    if meta is not None and (meta.track_name or meta.track_display_name):
        candidates = library.list_laps(track=meta.track_name or meta.track_display_name)
    if not candidates:
        candidates = library.list_laps(reference_only=True) or library.list_laps()
    timed = [c for c in candidates if c.lap_time is not None]
    if not timed:
        raise SystemExit("no reference lap in the library — import one or pass --reference-id")
    best = min(timed, key=lambda r: r.lap_time)
    print(f"reference: {best.label()}")
    return library.get_lap(best.lap_id)


def _run_overlay(args, source_samples, reference: LapData) -> int:
    from PySide6.QtWidgets import QApplication

    from .window import OverlayWindow

    app = QApplication.instance() or QApplication([])
    comparison = LiveComparison(reference)
    window = OverlayWindow(comparison)
    window.show()

    updates = 0
    for sample in source_samples:
        state = comparison.feed(sample)
        if state is not None:
            window.update_state(state)
            updates += 1
            if args.max_updates and updates >= args.max_updates:
                break
        app.processEvents()
        if args.realtime:
            time.sleep(1.0 / 60.0)
    if args.max_updates:
        return 0
    return app.exec()


def _replay_samples(path: str):
    with IbtReader(path) as reader:
        names = [n for n in LIVE_CHANNELS if reader.has_channel(n)]
        channels = reader.get_channels(names)
        n = reader.record_count
    for i in range(n):
        yield {name: channels[name][i] for name in names}


def _live_samples(live: LiveTelemetry, rate_hz: float):
    interval = 1.0 / rate_hz
    while True:
        try:
            yield live.snapshot_live_channels()
        except RuntimeError:
            return
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iracing-overlay", description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify shared core and exit")
    parser.add_argument("--watch", action="store_true", help="print live speed/lap once per second")
    parser.add_argument("--replay", default=None, help="drive the overlay from a .ibt file")
    parser.add_argument("--realtime", action="store_true", help="replay at real speed")
    parser.add_argument("--library-dir", default=None, help="reference library root")
    parser.add_argument("--reference-id", type=int, default=None)
    parser.add_argument("--rate", type=float, default=30.0, help="live sampling rate [Hz]")
    parser.add_argument("--max-updates", type=int, default=0, help="stop after N updates (testing)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    print(f"iracing_core {iracing_core.__version__} loaded from {iracing_core.__file__}")
    if args.check:
        return 0

    if args.watch:
        live = LiveTelemetry()
        if not live.connect():
            print("iRacing is not running (or not on Windows) — nothing to overlay.")
            return 1
        try:
            while True:
                snap = live.snapshot(["Speed", "Lap", "LapDistPct"])
                print(f"lap {snap['Lap']}  {snap['Speed'] * 3.6:6.1f} km/h  "
                      f"{snap['LapDistPct'] * 100.0:5.1f} %")
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0
        finally:
            live.disconnect()

    from pathlib import Path

    library = LapLibrary(args.library_dir or Path.home() / ".iracing_analysis" / "library")

    if args.replay:
        with IbtReader(args.replay) as reader:
            meta = reader.meta
        reference = _resolve_reference(library, args, meta)
        return _run_overlay(args, _replay_samples(args.replay), reference)

    live = LiveTelemetry()
    print("waiting for iRacing…")
    while not live.connect():
        time.sleep(5.0)
    meta = live.session_meta()
    reference = _resolve_reference(library, args, meta)
    try:
        return _run_overlay(args, _live_samples(live, args.rate), reference)
    finally:
        live.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
