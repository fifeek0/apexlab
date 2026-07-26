"""``garage61-harvest`` — build the reference library from Garage 61 Pro.

Selects a diverse set of laps per (track, car) combination through the
official API — a couple of outright references plus a percentile spread of
the visible field — downloads their telemetry CSVs under the canonical
export filename (so the existing importer and its physical-consistency
validation apply unchanged), imports them into the lap library tagged with
car/track/discipline, and optionally emits fine-tuning summaries.

Visibility: the API sees your own laps, teammates' and *followed* drivers'
(``--drivers me,following``). Follow fast drivers in the Garage 61 app to
widen the net; global search requires application approval by Garage 61.

Everything is resumable: a manifest in the cache dir records downloaded,
imported and rejected laps by their ULID, so re-runs (even with a different
cache dir) never duplicate work or re-download known-bad telemetry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from iracing_core import LapLibrary
from iracing_core.garage61 import format_garage61_filename, import_garage61_csv

from .garage61_api import Garage61Client, Garage61Error

__all__ = [
    "HarvestConfig",
    "HarvestStats",
    "PickedLap",
    "canonical_filename",
    "filter_usable",
    "harvest",
    "infer_discipline",
    "main",
    "resolve_combos",
    "resolve_token",
    "select_laps",
]

log = logging.getLogger(__name__)

#: rough CSV size per second of lap time (from real exports), for --dry-run
_BYTES_PER_LAP_SECOND = 9_000

DISCIPLINES = ("road", "open_wheel", "oval", "dirt_road", "dirt_oval")

#: default location of the API token (one line, never committed anywhere)
TOKEN_FILE = Path.home() / ".iracing_analysis" / "g61_token"


def resolve_token(
    cli_token: str | None,
    env: dict | None = None,
    token_file: Path = TOKEN_FILE,
) -> str | None:
    """Token precedence: --token > $G61_TOKEN > ~/.iracing_analysis/g61_token."""
    if cli_token:
        return cli_token.strip()
    env_token = (env if env is not None else os.environ).get("G61_TOKEN", "").strip()
    if env_token:
        return env_token
    try:
        text = Path(token_file).read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickedLap:
    item: dict
    is_reference: bool


def filter_usable(items: list[dict]) -> list[dict]:
    """Laps worth downloading: clean, complete, with visible telemetry and
    no recording discontinuity (the listing flags spare us corrupt files)."""
    return [
        lap
        for lap in items
        if lap.get("clean")
        and not lap.get("incomplete")
        and not lap.get("missing")
        and not lap.get("discontinuity")
        and lap.get("canViewTelemetry")
        and lap.get("lapTime")
    ]


def select_laps(
    items: list[dict], refs: int = 2, per_combo: int = 12, window: float = 1.2
) -> list[PickedLap]:
    """``refs`` fastest laps + a percentile spread of the remaining field
    (within ``window``×P1), at most ``per_combo`` laps, sorted by time."""
    if not items:
        return []
    ordered = sorted(items, key=lambda lap: lap["lapTime"])
    cutoff = ordered[0]["lapTime"] * window
    ordered = [lap for lap in ordered if lap["lapTime"] <= cutoff]

    refs = min(refs, len(ordered))
    picked = [PickedLap(lap, True) for lap in ordered[:refs]]

    rest = ordered[refs:]
    need = min(per_combo - refs, len(rest))
    if need > 0:
        idx = np.unique(np.linspace(0, len(rest) - 1, need).round().astype(int))
        picked.extend(PickedLap(rest[i], False) for i in idx)
    return picked


def _driver_name(item: dict) -> str:
    """The real API sends firstName/lastName; docs (and older payloads) name."""
    driver = item.get("driver") or {}
    if driver.get("name"):
        return str(driver["name"])
    full = f"{driver.get('firstName', '')} {driver.get('lastName', '')}".strip()
    return full or str(driver.get("slug", "unknown"))


def canonical_filename(item: dict) -> str:
    track = item.get("track") or {}
    label = track.get("name", "unknown")
    if track.get("variant"):
        label = f"{label} ({track['variant']})"
    car = (item.get("car") or {}).get("name", "unknown")
    return format_garage61_filename(_driver_name(item), car, label, item["lapTime"], item["id"])


def infer_discipline(car_group_name: str | None, track_name: str) -> str:
    """Best-effort discipline from the car group and track names."""
    group = (car_group_name or "").lower()
    track = (track_name or "").lower()
    if "rallycross" in group or "rallycross" in track:
        return "dirt_road"  # mixed-surface; closest bucket we have
    dirt = "dirt" in group or "dirt" in track
    # road courses hosted at speedway facilities (roval, road course, GP
    # layouts) are road, not oval
    road_hints = ("road course", "roval", "grand prix", "street", "infield")
    speedway_oval = "speedway" in track and not any(h in track for h in road_hints)
    oval = (
        "oval" in group
        or speedway_oval
        or "nascar" in group
        or group in ("legends", "late model", "sprint car")
    )
    if dirt and oval:
        return "dirt_oval"
    if dirt:
        return "dirt_road"
    if oval:
        return "oval"
    if any(word in group for word in ("formula", "open wheel", "openwheel", "indy")):
        return "open_wheel"
    return "road"


# ---------------------------------------------------------------------------
# harvesting
# ---------------------------------------------------------------------------


@dataclass
class HarvestConfig:
    cache_dir: Path
    per_combo: int = 12
    refs: int = 2
    window: float = 1.2
    max_laps: int = 500
    drivers: tuple[str, ...] = ("me", "following")
    dry_run: bool = False
    tags: tuple[str, ...] = ("g61", "harvest")


@dataclass
class HarvestStats:
    listed: int = 0
    selected: int = 0
    downloaded: int = 0
    imported: int = 0
    rejected: int = 0
    skipped: int = 0
    errors: int = 0


class _Manifest:
    """ULID → status record in the cache dir; makes every run resumable and
    keeps physically-rejected laps as a terminal state (no re-downloads)."""

    def __init__(self, cache_dir: Path):
        self.path = Path(cache_dir) / "manifest.json"
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("unreadable manifest %s — starting fresh", self.path)

    def mark(self, lap_id: str, status: str, **extra) -> None:
        self.entries[lap_id] = {"status": status, **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")

    def known(self, lap_id: str) -> bool:
        return lap_id in self.entries


def _known_ulids_in_library(library: LapLibrary) -> set[str]:
    """ULIDs already imported, recovered from source filenames — makes the
    dedupe survive cache-dir moves."""
    from iracing_core.garage61 import parse_garage61_filename

    ulids = set()
    for record in library.list_laps():
        info = parse_garage61_filename(Path(record.source_file or "").name)
        if info.get("lap_id"):
            ulids.add(info["lap_id"])
    return ulids


def harvest(
    client: Garage61Client,
    library: LapLibrary,
    combos: list[tuple[int, str, int, str, str | None]],
    config: HarvestConfig,
) -> HarvestStats:
    """Run the harvest over ``combos`` = (track_id, track_label, car_id,
    car_label, car_group_name)."""
    stats = HarvestStats()
    manifest = _Manifest(config.cache_dir)
    seen = _known_ulids_in_library(library) | set(manifest.entries)
    budget = config.max_laps

    for track_id, track_label, car_id, car_label, group_name in combos:
        try:
            items = _list_combo(client, track_id, car_id, config.drivers)
        except Garage61Error as exc:
            log.error("listing failed for %s / %s: %s", track_label, car_label, exc)
            stats.errors += 1
            continue
        stats.listed += len(items)
        picked = select_laps(
            filter_usable(items), refs=config.refs,
            per_combo=config.per_combo, window=config.window,
        )
        stats.selected += len(picked)
        discipline = infer_discipline(group_name, track_label)

        if config.dry_run:
            est = sum(p.item["lapTime"] for p in picked) * _BYTES_PER_LAP_SECOND
            print(f"\n{track_label} × {car_label}  [{discipline}]  (~{est / 1e6:.0f} MB):")
            for p in picked:
                item = p.item
                marker = "REF " if p.is_reference else "    "
                print(
                    f"  {marker}{item['lapTime']:9.3f}s  "
                    f"iR {item.get('driverRating') or '----'}  "
                    f"{_driver_name(item)}"
                )
            continue

        for pick in picked:
            lap_id = pick.item["id"]
            if lap_id in seen:
                stats.skipped += 1
                continue
            if budget <= 0:
                log.warning("--max-laps budget reached; stopping")
                return stats
            budget -= 1

            path = Path(config.cache_dir) / canonical_filename(pick.item)
            try:
                if not path.exists():
                    payload = client.download_lap_csv(lap_id)
                    part = path.with_suffix(".part")
                    part.parent.mkdir(parents=True, exist_ok=True)
                    part.write_bytes(payload)
                    os.replace(part, path)
                    stats.downloaded += 1
            except Garage61Error as exc:
                log.warning("download failed for %s: %s", lap_id, exc)
                if exc.status in (402, 403):
                    # driver allows viewing but not exporting: terminal state
                    manifest.mark(lap_id, "denied", reason=f"HTTP {exc.status}")
                    seen.add(lap_id)
                stats.errors += 1
                continue

            tags = (
                *config.tags,
                f"discipline:{discipline}",
                _slug(car_label),
                _slug(track_label),
            )
            try:
                import_garage61_csv(library, path, tags=tags)
            except ValueError as exc:
                log.warning("rejected %s: %s", path.name, exc)
                manifest.mark(lap_id, "rejected", reason=str(exc)[:200], path=str(path))
                stats.rejected += 1
                seen.add(lap_id)
                continue
            manifest.mark(lap_id, "imported", path=str(path))
            seen.add(lap_id)
            stats.imported += 1

        if not config.dry_run and picked:
            # reconcile against the labels as they land in the library
            # (canonical filenames carry the track variant, combos may not);
            # group-level combos mix car makes -> reconcile each one
            pairs = set()
            for pick in picked:
                item = pick.item
                track = item.get("track") or {}
                full_label = track.get("name", track_label)
                if track.get("variant"):
                    full_label = f"{full_label} ({track['variant']})"
                pairs.add((full_label, (item.get("car") or {}).get("name", car_label)))
            for full_label, car_name in sorted(pairs):
                _reconcile_references(library, full_label, car_name)
    return stats


def _list_combo(
    client: Garage61Client, track_id: int, car_id: int, drivers: tuple[str, ...]
) -> list[dict]:
    """Merge the two visibility scopes: passing ``drivers`` to the API
    *replaces* the default own+teammates scope, so 'me' means one query with
    no driver filter (own + all teammates) and 'following' a second one."""
    base = dict(tracks=[track_id], cars=[car_id], group="driver",
                lapTypes=[1], seeTelemetry=True)
    merged: dict[str, dict] = {}
    if "me" in drivers or not drivers:
        for item in client.find_laps(**base):
            merged[item["id"]] = item
    if "following" in drivers:
        for item in client.find_laps(drivers=["following"], **base):
            merged[item["id"]] = item
    return list(merged.values())


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")[:40]


def _reconcile_references(library: LapLibrary, track_label: str, car_label: str) -> None:
    """Exactly one reference per (track, car): the fastest imported lap —
    re-imports never update flags on their own."""
    records = [
        r for r in library.list_laps(track=track_label, car=car_label)
        if r.lap_time is not None
    ]
    if not records:
        return
    fastest = min(records, key=lambda r: r.lap_time)
    for record in records:
        should_be = record.lap_id == fastest.lap_id
        if record.is_reference != should_be:
            library.set_reference(record.lap_id, should_be)


# ---------------------------------------------------------------------------
# name resolution + CLI
# ---------------------------------------------------------------------------


def _resolve(items: list[dict], term: str, kind: str, label=lambda x: x["name"]) -> dict:
    lowered = term.lower()
    exact = [x for x in items if label(x).lower() == lowered]
    if exact:
        return exact[0]
    matches = [x for x in items if lowered in label(x).lower()]
    if not matches:
        raise ValueError(f"no {kind} matches {term!r}")
    if len(matches) > 1:
        names = ", ".join(sorted(label(m) for m in matches)[:8])
        raise ValueError(f"{kind} {term!r} is ambiguous: {names}")
    return matches[0]


def _track_label(track: dict) -> str:
    if track.get("variant"):
        return f"{track['name']} ({track['variant']})"
    return track["name"]


def resolve_combos(
    client: Garage61Client,
    car_terms: list[str],
    group_terms: list[str],
    track_terms: list[str],
) -> list[tuple[int, str, int, str, str | None]]:
    """(track_id, track_label, car_id, car_label, group_name) per unique pair."""
    tracks = [
        _resolve(client.tracks(), term, "track", label=_track_label)
        for term in track_terms
    ]
    cars_by_id = {c["id"]: c for c in client.cars()}

    # --car = one combo per car; --car-group = ONE group-level combo per
    # track (negative id, the API mixes all the group's cars in one search)
    selections: list[tuple[int, str, str | None]] = []
    for term in car_terms:
        car = _resolve(list(cars_by_id.values()), term, "car")
        selections.append((car["id"], car["name"], None))
    for term in group_terms:
        group = _resolve(client.car_groups(), term, "car group")
        selections.append((-group["id"], group["name"], group["name"]))

    if not selections:
        raise ValueError("no cars selected — pass --car and/or --car-group")

    combos = []
    for track in tracks:
        for car_id, car_label, group_name in selections:
            combos.append((track["id"], _track_label(track), car_id, car_label, group_name))
    return combos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="garage61-harvest",
        description="Harvest reference laps from Garage 61 (official API, Pro "
        "recommended) into the local lap library.",
    )
    parser.add_argument("--token", default=None,
                        help="personal access token (default: $G61_TOKEN or ~/.iracing_analysis/g61_token)")
    parser.add_argument("--car", action="append", default=[], help="car name (repeatable)")
    parser.add_argument("--car-group", action="append", default=[],
                        help="car group/category name, e.g. 'Sports Car' (repeatable)")
    parser.add_argument("--track", action="append", default=[], required=False,
                        help="track name (repeatable)")
    parser.add_argument("--drivers", default="me,following",
                        help="lap visibility scope (default: me,following)")
    parser.add_argument("--refs", type=int, default=2)
    parser.add_argument("--per-combo", type=int, default=12)
    parser.add_argument("--window", type=float, default=1.2,
                        help="lap-time cutoff as a multiple of P1 (default 1.2)")
    parser.add_argument("--max-laps", type=int, default=500)
    parser.add_argument("--rate", type=float, default=2.0, help="seconds between API calls")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".iracing_analysis" / "g61_cache"))
    parser.add_argument("--library-dir", default=str(Path.home() / ".iracing_analysis" / "library"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export", default=None, help="write fine-tuning summaries JSONL when done")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    token = resolve_token(args.token)
    if not token:
        print(f"error: no token — paste it into {TOKEN_FILE}, or pass --token / set G61_TOKEN "
              "(garage61.net → My applications)")
        return 1
    if not args.track:
        print("error: pass at least one --track")
        return 1

    client = Garage61Client(token=token, min_interval_s=args.rate)
    library = LapLibrary(args.library_dir)
    try:
        combos = resolve_combos(client, args.car, args.car_group, args.track)
    except (ValueError, Garage61Error) as exc:
        print(f"error: {exc}")
        return 1

    config = HarvestConfig(
        cache_dir=Path(args.cache_dir),
        per_combo=args.per_combo,
        refs=args.refs,
        window=args.window,
        max_laps=args.max_laps,
        drivers=tuple(t.strip() for t in args.drivers.split(",") if t.strip()),
        dry_run=args.dry_run,
    )
    stats = harvest(client, library, combos, config)
    print(
        f"\nlisted {stats.listed}, selected {stats.selected}, downloaded {stats.downloaded}, "
        f"imported {stats.imported}, rejected {stats.rejected}, skipped {stats.skipped}, "
        f"errors {stats.errors}"
    )

    if args.export and not args.dry_run:
        from .export_summaries import export_summaries

        ex = export_summaries(telemetry_dirs=[], library_dirs=[Path(args.library_dir)],
                              out_path=Path(args.export))
        print(f"exported {ex.written} summaries -> {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
