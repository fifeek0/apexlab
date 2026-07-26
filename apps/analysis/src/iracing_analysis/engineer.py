"""Live race engineer: lap-by-lap 'where you lose time' radio updates.

Our Crew-Chief-style feature: while you drive (or replay a stint), every
completed lap is aligned against a reference lap from the library, the
per-corner losses are attributed, and a short engineer message is produced
— printed, optionally spoken (pyttsx3), and optionally phrased by the local
LLM as a natural radio line.

Run it with ``iracing-engineer`` (see ``--help``): live against the sim on
Windows, or ``--replay stint.ibt`` anywhere.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from iracing_core import LapData, LapLibrary, LiveLapRecorder, LiveTelemetry
from iracing_core.live import LIVE_CHANNELS

from .analysis.corners import CornerComparison, analyze_corners, rank_by_time_lost
from .analysis.workspace import Workspace
from .insights.base import InsightProvider
from .voice import BaseSpeaker, create_speaker

__all__ = ["CornerLoss", "EngineerUpdate", "RaceEngineer", "main"]

log = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    mins, secs = divmod(seconds, 60.0)
    return f"{int(mins)}:{secs:06.3f}"


@dataclass(frozen=True)
class CornerLoss:
    label: str
    time_lost: float
    why: str


@dataclass(frozen=True)
class EngineerUpdate:
    lap_number: int
    lap_time: float
    delta_total: float  # + slower than reference
    top_losses: list[CornerLoss]
    gains: list[CornerLoss]
    message: str
    radio_line: str | None = None  # LLM-phrased, when a provider is set


def _short_why(comp: CornerComparison, language: str) -> str:
    """One compact cue per corner, derived from the measured differences."""
    m, r = comp.metrics, comp.ref_metrics
    pl = language == "pl"

    dv = (r.min_speed - m.min_speed) * 3.6
    if dv > 2.0:
        return f"apeks {dv:.0f} km/h za wolny" if pl else f"apex {dv:.0f} km/h slow"
    if m.throttle_full_m is not None and r.throttle_full_m is not None:
        late = m.throttle_full_m - r.throttle_full_m
        if late > 8.0:
            return f"gaz {late:.0f} m za późno" if pl else f"throttle {late:.0f} m late"
    if m.braking_point_m is not None and r.braking_point_m is not None:
        early = r.braking_point_m - m.braking_point_m
        if early > 5.0:
            return f"hamowanie {early:.0f} m za wcześnie" if pl else f"braking {early:.0f} m early"
    trail = m.trail_brake_overlap_m - r.trail_brake_overlap_m
    if trail > 10.0:
        return "za długo na hamulcu w skręcie" if pl else "too much trail brake"
    if comp.delta_carry > max(comp.delta_entry, comp.delta_exit, 0.0):
        return "słabe wyjście, strata na prostej" if pl else "slow exit, losing on the straight"
    return "drobne różnice na całej długości" if pl else "small differences throughout"


class RaceEngineer:
    """Compares each finished lap against a fixed reference lap."""

    def __init__(
        self,
        reference: LapData,
        language: str = "pl",
        provider: InsightProvider | None = None,
        top_n: int = 3,
    ):
        if reference.lap_time is None:
            raise ValueError("reference lap has no lap time")
        self.reference = reference
        self.language = language
        self.provider = provider
        self.top_n = top_n

    # -- per-lap analysis ---------------------------------------------------

    def on_lap(self, lap: LapData) -> EngineerUpdate | None:
        if lap.lap_time is None:
            return None
        try:
            workspace = Workspace([self.reference, lap])
            ranked = rank_by_time_lost(analyze_corners(workspace, 1))
        except Exception as exc:
            log.warning("lap analysis failed: %s", exc)
            return None

        losses = [
            CornerLoss(c.corner.label(), c.time_lost, _short_why(c, self.language))
            for c in ranked
            if c.time_lost > 0.05
        ][: self.top_n]
        gains = [
            CornerLoss(c.corner.label(), c.time_lost, "")
            for c in reversed(ranked)
            if c.time_lost < -0.05
        ][:2]

        delta_total = lap.lap_time - self.reference.lap_time
        update = EngineerUpdate(
            lap_number=lap.lap_number,
            lap_time=lap.lap_time,
            delta_total=delta_total,
            top_losses=losses,
            gains=gains,
            message=self._message(lap, delta_total, losses, gains),
        )
        radio = self._radio_line(update)
        if radio:
            update = EngineerUpdate(**{**update.__dict__, "radio_line": radio})
        return update

    # -- phrasing ---------------------------------------------------------------

    def _message(
        self,
        lap: LapData,
        delta: float,
        losses: list[CornerLoss],
        gains: list[CornerLoss],
    ) -> str:
        pl = self.language == "pl"
        t = _fmt_time(lap.lap_time)
        if delta < -0.005:
            head = (
                f"Okrążenie {t}, {-delta:.2f} szybciej od referencji!"
                if pl
                else f"Lap {t}, {-delta:.2f} faster than the reference!"
            )
        else:
            head = (
                f"Okrążenie {t}, {delta:+.2f} do referencji."
                if pl
                else f"Lap {t}, {delta:+.2f}s to the reference."
            )
        parts = [head]
        if losses:
            worst = ", ".join(
                f"{c.label} {c.time_lost:+.2f} ({c.why})" for c in losses
            )
            parts.append((f"Najwięcej tracisz: {worst}." if pl else f"Biggest losses: {worst}."))
            parts.append(
                f"Skup się na {losses[0].label}." if pl else f"Focus on {losses[0].label}."
            )
        if gains:
            good = ", ".join(f"{c.label} {c.time_lost:+.2f}" for c in gains)
            parts.append(f"Zyskujesz w: {good}." if pl else f"Gaining in: {good}.")
        return " ".join(parts)

    def _radio_line(self, update: EngineerUpdate) -> str | None:
        if self.provider is None:
            return None
        summary = {
            "task": "radio_message",
            "language": "Polish" if self.language == "pl" else "English",
            "lap_time_s": round(update.lap_time, 3),
            "delta_to_reference_s": round(update.delta_total, 3),
            "corners": [
                {"corner": c.label, "time_lost_s": round(c.time_lost, 2), "why": c.why}
                for c in update.top_losses
            ],
            "gains": [
                {"corner": c.label, "time_gained_s": round(-c.time_lost, 2)}
                for c in update.gains
            ],
        }
        try:
            result = self.provider.generate(summary)
        except Exception as exc:  # provider bugs must not kill the pit wall
            log.warning("radio line generation failed: %s", exc)
            return None
        return result.text.strip() if result.ok and result.text.strip() else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_reference(library: LapLibrary, args, meta) -> LapData:
    if args.reference_id is not None:
        return library.get_lap(args.reference_id)

    candidates = []
    if meta is not None and meta.track_name:
        candidates = library.list_laps(track=meta.track_name)
    if not candidates and meta is not None and meta.track_display_name:
        candidates = library.list_laps(track=meta.track_display_name)
    if not candidates and meta is not None and meta.car_path:
        candidates = library.list_laps(car=meta.car_path)
        if candidates:
            log.warning("no track match in the library; falling back to car match")
    if not candidates:
        candidates = library.list_laps(reference_only=True)
    timed = [c for c in candidates if c.lap_time is not None]
    if not timed:
        raise SystemExit(
            "no usable reference lap in the library — import one or pass --reference-id"
        )
    best = min(timed, key=lambda r: r.lap_time)
    log.info("reference: %s", best.label())
    return library.get_lap(best.lap_id)


def _replay(args, engineer: RaceEngineer, speaker: BaseSpeaker) -> int:
    from iracing_core import IbtReader

    recorder = LiveLapRecorder()
    with IbtReader(args.replay) as reader:
        names = [n for n in LIVE_CHANNELS if reader.has_channel(n)]
        channels = reader.get_channels(names)
        n = reader.record_count
        recorder.meta = reader.meta

    def handle(lap: LapData | None) -> None:
        if lap is None:
            return
        update = engineer.on_lap(lap)
        if update is None:
            print(f"[lap {lap.lap_number}] no time (out/in lap?) — skipped")
            return
        line = update.radio_line or update.message
        print(f"[lap {update.lap_number}] {line}")
        speaker.say(line)

    for i in range(n):
        handle(recorder.feed({name: channels[name][i] for name in names}))
        if args.realtime:
            time.sleep(1.0 / 60.0)
    handle(recorder.finish())
    return 0


def _live(args, engineer_factory, speaker: BaseSpeaker, library: LapLibrary) -> int:
    live = LiveTelemetry()
    print("waiting for iRacing…")
    while not live.connect():
        time.sleep(5.0)
    meta = live.session_meta()
    engineer = engineer_factory(meta)
    print(f"connected: {meta.track_display_name or meta.track_name} — reference "
          f"{_fmt_time(engineer.reference.lap_time)}")
    recorder = LiveLapRecorder(meta=meta)
    interval = 1.0 / args.rate
    try:
        while True:
            try:
                sample = live.snapshot_live_channels()
            except RuntimeError:
                print("sim gone; waiting…")
                live.disconnect()
                while not live.connect():
                    time.sleep(5.0)
                continue
            if sample.get("Lap") is not None:
                lap = recorder.feed(sample)
                if lap is not None:
                    update = engineer.on_lap(lap)
                    if update is not None:
                        line = update.radio_line or update.message
                        print(f"[lap {update.lap_number}] {line}")
                        speaker.say(line)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("pit wall signing off")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="iracing-engineer",
        description="Live lap-by-lap race engineer: compares every lap against "
        "a reference from your library and tells you where you lose time.",
    )
    parser.add_argument("--library-dir", default=str(Path.home() / ".iracing_analysis" / "library"))
    parser.add_argument("--reference-id", type=int, default=None, help="library lap id to compare against")
    parser.add_argument("--language", choices=("pl", "en"), default="pl")
    parser.add_argument("--llm", action="store_true", help="phrase updates via the configured AI endpoint")
    parser.add_argument("--tts", action="store_true", help="speak updates aloud")
    parser.add_argument(
        "--tts-engine", choices=("auto", "piper", "system", "off"), default="auto",
        help="voice backend: piper (neural, needs a downloaded voice), system (pyttsx3)",
    )
    parser.add_argument("--piper-model", default=None, help="path to a piper .onnx voice model")
    parser.add_argument("--top", type=int, default=3, help="corners per update")
    parser.add_argument("--rate", type=float, default=30.0, help="live sampling rate [Hz]")
    parser.add_argument("--replay", default=None, help="replay a .ibt file instead of the live sim")
    parser.add_argument("--realtime", action="store_true", help="replay at real speed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    library = LapLibrary(args.library_dir)
    speaker = create_speaker(
        args.tts_engine if args.tts else "off", piper_model=args.piper_model
    )

    provider = None
    if args.llm:
        from .config import load_config
        from .insights import create_provider

        provider = create_provider(load_config().ai)

    def engineer_for(meta) -> RaceEngineer:
        reference = _resolve_reference(library, args, meta)
        return RaceEngineer(reference, language=args.language, provider=provider, top_n=args.top)

    if args.replay:
        from iracing_core import IbtReader

        with IbtReader(args.replay) as reader:
            meta = reader.meta
        return _replay(args, engineer_for(meta), speaker)
    return _live(args, engineer_for, speaker, library)


if __name__ == "__main__":
    raise SystemExit(main())
