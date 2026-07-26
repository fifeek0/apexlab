"""Export analysis summaries from REAL telemetry as fine-tuning inputs.

Walks telemetry folders (``.ibt``) and reference libraries, groups clean
laps by (track, car), pairs every lap with the fastest one of its group and
emits one JSON summary per pair — the same ``build_summary`` payload the
app sends to the LLM. The fine-tuning pipeline then feeds these to the
teacher model exactly like the synthetic scenarios, adding real-world
diversity (real tracks and corner counts, real mistakes, tyre signals from
``.ibt``) that synthetic data can't provide.

Usage::

    python -m iracing_analysis.export_summaries \
        --telemetry-dir ~/Documents/iRacing/telemetry \
        --library-dir ~/.iracing_analysis/library \
        --out racing_real_summaries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from iracing_core import IbtReader, LapData, LapLibrary, extract_laps, scan_telemetry_dir

from .analysis.workspace import Workspace
from .insights.summary import build_summary

__all__ = ["ExportStats", "export_summaries", "main"]

log = logging.getLogger(__name__)


@dataclass
class ExportStats:
    written: int = 0
    skipped_groups: int = 0
    failed: int = 0


def _group_key(lap: LapData) -> tuple[str, str]:
    meta = lap.meta
    track = (meta.track_name or meta.track_display_name) if meta else ""
    car = (meta.car_path or meta.car_screen_name) if meta else ""
    return (track.lower(), car.lower())


def _lap_id(lap: LapData) -> str:
    src = str(lap.meta.path) if lap.meta else "?"
    return f"{src}::{lap.lap_number}"


def _collect_laps(
    telemetry_dirs: list[Path], library_dirs: list[Path]
) -> tuple[dict[tuple[str, str], list[LapData]], dict[str, tuple[str, ...]]]:
    groups: dict[tuple[str, str], dict[str, LapData]] = {}
    tags_by_lap: dict[str, tuple[str, ...]] = {}

    def add(lap: LapData, tags: tuple[str, ...] = ()) -> None:
        if not (lap.is_clean and lap.lap_time):
            return
        groups.setdefault(_group_key(lap), {})[_lap_id(lap)] = lap
        if tags:
            tags_by_lap[_lap_id(lap)] = tags

    for root in telemetry_dirs:
        for session_group in scan_telemetry_dir(root):
            for meta in session_group.files:
                try:
                    with IbtReader(meta.path) as reader:
                        for lap in extract_laps(reader):
                            add(lap)
                except Exception as exc:
                    log.warning("skipping %s: %s", meta.path, exc)

    for root in library_dirs:
        library = LapLibrary(root)
        for record in library.list_laps():
            try:
                add(library.get_lap(record.lap_id), tags=record.tags)
            except Exception as exc:
                log.warning("skipping library lap %s: %s", record.lap_id, exc)

    return {key: list(laps.values()) for key, laps in groups.items()}, tags_by_lap


def _discipline_from_tags(tags: tuple[str, ...]) -> str:
    for tag in tags:
        if tag.startswith("discipline:"):
            return tag.split(":", 1)[1]
    return "road"


def export_summaries(
    telemetry_dirs: list[Path],
    library_dirs: list[Path],
    out_path: Path,
    max_pairs_per_group: int = 25,
) -> ExportStats:
    """Build (lap vs group-best) summaries and append them to ``out_path``."""
    stats = ExportStats()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # append-only with dedupe: pair ids already in the file are skipped
    existing_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(json.loads(line).get("id", ""))
            except json.JSONDecodeError:
                continue

    groups, tags_by_lap = _collect_laps(
        [Path(p) for p in telemetry_dirs], [Path(p) for p in library_dirs]
    )
    with open(out_path, "a", encoding="utf-8") as out:
        for (track, car), laps in sorted(groups.items()):
            if len(laps) < 2:
                stats.skipped_groups += 1
                continue
            laps.sort(key=lambda lap: lap.lap_time)
            reference, rest = laps[0], laps[1 : 1 + max_pairs_per_group]
            for lap in rest:
                pair_id = f"real::{_lap_id(lap)}::vs::{_lap_id(reference)}"
                if pair_id in existing_ids:
                    continue
                discipline = _discipline_from_tags(tags_by_lap.get(_lap_id(lap), ()))
                try:
                    workspace = Workspace([reference, lap])
                    summary = build_summary(workspace, lap_index=1)
                except Exception as exc:
                    log.warning("pair failed (%s vs %s): %s", _lap_id(lap), _lap_id(reference), exc)
                    stats.failed += 1
                    continue
                summary["session"]["discipline"] = discipline
                row = {
                    "id": pair_id,
                    "source": "real",
                    "track": summary["session"]["track"] or track,
                    "car": summary["session"]["car"] or car,
                    "discipline": discipline,
                    "summary": summary,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats.written += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m iracing_analysis.export_summaries", description=__doc__
    )
    parser.add_argument("--telemetry-dir", action="append", default=[],
                        help="folder with .ibt files (repeatable)")
    parser.add_argument("--library-dir", action="append", default=[],
                        help="reference library root (repeatable)")
    parser.add_argument("--out", required=True, help="output JSONL path (appended)")
    parser.add_argument("--max-pairs-per-group", type=int, default=25)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.telemetry_dir and not args.library_dir:
        print("error: pass at least one --telemetry-dir or --library-dir")
        return 1

    stats = export_summaries(
        telemetry_dirs=args.telemetry_dir,
        library_dirs=args.library_dir,
        out_path=Path(args.out),
        max_pairs_per_group=args.max_pairs_per_group,
    )
    print(
        f"written {stats.written} summaries "
        f"({stats.skipped_groups} groups skipped, {stats.failed} pairs failed) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
