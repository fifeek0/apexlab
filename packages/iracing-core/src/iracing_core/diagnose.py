"""One-command validation of a real ``.ibt`` file against this toolchain.

Usage::

    python -m iracing_core.diagnose "Documents/iRacing/telemetry/car/stint.ibt"

Prints file structure, YAML metadata, channel coverage, the lap table and a
delta-engine cross-check (delta at S/F vs lap-time difference). Meant as the
first thing to run on telemetry from a new source (own PC, teammate, pro).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .alignment import align_laps, delta_time
from .ibt import IbtReader
from .sessions import DEFAULT_CHANNELS, extract_laps

__all__ = ["main", "diagnose"]


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.---"
    mins, secs = divmod(seconds, 60.0)
    return f"{int(mins)}:{secs:06.3f}"


def diagnose(path: Path, out=print) -> int:
    with IbtReader(path) as reader:
        meta = reader.meta
        n = reader.record_count
        rate = reader.tick_rate
        names = reader.channel_names

        out(f"File     : {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        out(f"Format   : {rate} Hz, {n} records, {n / max(rate, 1) / 60.0:.1f} min")
        track = f"Track    : {meta.track_display_name or meta.track_name} ({meta.track_config or 'default'})"
        if meta.track_length_m:
            track += f", {meta.track_length_m:.0f} m"
        out(track)
        out(f"Car      : {meta.car_screen_name} ({meta.car_path})")
        out(f"Driver   : {meta.driver_name}")
        out(f"Session  : {meta.session_type or '?'}  id={meta.session_id} "
            f"sub={meta.subsession_id}  start={meta.session_start_date}")
        out(f"Sectors  : {['%.3f' % p for p in meta.sector_starts_pct] or 'none in YAML'}")

        present = [c for c in DEFAULT_CHANNELS if reader.has_channel(c)]
        missing = [c for c in DEFAULT_CHANNELS if not reader.has_channel(c)]
        out(f"Channels : {len(names)} total; core loaded: {len(present)}/{len(DEFAULT_CHANNELS)}")
        if missing:
            out(f"  missing (ok if car lacks them): {', '.join(missing)}")

        laps = extract_laps(reader)

    out(f"Laps     : {len(laps)}")
    for lap in laps:
        flags = [
            name
            for name, is_set in (
                ("out", lap.is_out_lap),
                ("in", lap.is_in_lap),
                ("pit", lap.touched_pits and not (lap.is_out_lap or lap.is_in_lap)),
                ("partial", not lap.is_complete),
                ("clean", lap.is_clean),
            )
            if is_set
        ]
        out(f"  L{lap.lap_number:<3} {_fmt_time(lap.lap_time)}  "
            f"{lap.n_samples:>6} samples  {','.join(flags)}")

    clean = sorted(
        (lap for lap in laps if lap.is_clean and lap.lap_time),
        key=lambda lap: lap.lap_time,
    )
    if len(clean) >= 2:
        ref, other = clean[0], clean[1]
        aligned = align_laps([ref, other])
        d_finish = float(delta_time(aligned.laps[1], aligned.laps[0])[-1])
        expected = other.lap_time - ref.lap_time
        ok = abs(d_finish - expected) < 0.005
        out(f"Delta check: L{other.lap_number} vs L{ref.lap_number}: "
            f"S/F delta {d_finish:+.4f} s vs lap-time diff {expected:+.4f} s "
            f"-> {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 2
    out("Delta check: skipped (needs >= 2 clean laps)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m iracing_core.diagnose", description=__doc__
    )
    parser.add_argument("ibt_file", help="path to a .ibt telemetry file")
    args = parser.parse_args(argv)

    path = Path(args.ibt_file)
    try:
        return diagnose(path)
    except Exception as exc:
        print(f"error: cannot analyse {path}: {exc}")
        traceback.print_exc(file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
