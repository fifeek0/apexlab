"""CSV-driven harvest campaign.

The dataset spec lives in ``iracing_telemetry_dataset.csv``
(Seria, Klasa_Samochodu, Model_Samochodu, Tor, Konfiguracja) — edit the CSV
to grow the dataset. This script resolves cars/tracks to exact Garage 61
entities (with a mapping report), harvests every unique combination and
exports fine-tuning summaries. Fully resumable.

Run:  .venv/bin/python tools/harvest_campaign.py [--dry-run] [dataset.csv]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from iracing_core import LapLibrary

from iracing_analysis.export_summaries import export_summaries
from iracing_analysis.garage61_api import Garage61Client
from iracing_analysis.harvest import (
    HarvestConfig,
    HarvestStats,
    harvest,
    resolve_token,
)

DATASET = Path(__file__).resolve().parent.parent / "iracing_telemetry_dataset.csv"

#: name fragments that differ between the CSV and Garage 61's car names
#: (keys are matched AFTER normalization: lowercase, punctuation -> space)
CAR_ALIASES = {
    "mazda mx 5 cup": "global mazda mx 5 cup",
    "nextgen": "next gen",
    "dallara sf23 sf 19": "sf23",
        "mercedes amg gt3 evo": "mercedes amg gt3 2020",
    "lamborghini huracan gt3 evo": "lamborghini huracán gt3 evo",
    "ferrari 296 gt3": "ferrari 296 gt3",
    "nascar nextgen": "nascar cup series next gen",
    "dirt sprint car 410 cased": "dirt sprint car",
    "ford fiesta rs wrc rx": "ford fiesta rs",
    "legends ford 34 coupe": "legends ford",
    "dallara sf23": "super formula sf23",
    "mercedes amg f1 w13 e performance": "w13",
    "subaru wrx sti rx": "subaru wrx",
    "st petersburg street course": "st petersburg",
    # tracks whose iRacing/G61 naming differs from the series calendars
    "bristol motor speedway oval": "bristol motor speedway single pit roads",
    "daytona international speedway rallycross long": "daytona rallycross and dirt road rallycross long",
    "daytona international speedway short rx": "daytona rallycross and dirt road rallycross short",
    "autodromo jose carlos pace": "autódromo josé carlos pace",
    "brands hatch rallycross": "brands hatch rallycross",
    "barcelona catalunya rallycross": "barcelona rallycross",
    "phoenix raceway rallycross": "phoenix raceway rallycross",
    "hockenheimring baden wurttemberg": "hockenheimring baden württemberg",
    "nurburgring grand prix strecke": "nürburgring grand prix strecke",
    "nurburgring nordschleife": "nürburgring nordschleife",
    "autodromo nazionale monza historic without chicanes": "autodromo nazionale monza grand prix without chicanes",
    "michelin raceway road atlanta": "road atlanta",
    "twin ring motegi": "mobility resort motegi",
    "porsche 718 cayman gt4 clubsport": "porsche 718 cayman gt4",
    "imola circuit": "autodromo internazionale enzo dino ferrari",
    "brand hatch": "brands hatch",
    "holden commodore zb supercar": "holden zb commodore",
    "ford mustang gt supercar": "ford mustang supercars",
    "oschersleben": "motorsport arena oschersleben",
}


def _norm(text: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(cleaned.split())


def _word_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) > 3 and len(b) > 3:
        return a.startswith(b) or b.startswith(a)
    return False


def match_entity(term: str, items: list[dict], label_of) -> dict | None:
    """All words of ``term`` must match label words; fewest unmatched label
    words wins, then the shortest label."""
    lowered = _norm(term).strip()
    for alias, replacement in CAR_ALIASES.items():
        if alias in lowered:
            lowered = lowered.replace(alias, replacement)
    words = [w for w in lowered.split() if len(w) > 1]
    candidates = []
    for item in items:
        label_words = [w for w in _norm(label_of(item)).split() if len(w) > 1]
        if not words or not all(any(_word_match(w, lw) for lw in label_words) for w in words):
            continue
        extra = sum(1 for lw in label_words if not any(_word_match(w, lw) for w in words))
        candidates.append((extra, len(label_words), item))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    map_only = "--map-only" in sys.argv
    dataset = Path(args[0]) if args else DATASET

    token = resolve_token(None)
    if not token:
        print("error: no token")
        return 1
    client = Garage61Client(token=token, min_interval_s=2.0)
    library = LapLibrary(Path.home() / ".iracing_analysis" / "library")

    tracks = client.tracks()
    track_label = lambda t: f"{t['name']} ({t['variant']})" if t.get("variant") else t["name"]  # noqa: E731
    cars = client.cars()

    rows = list(csv.DictReader(open(dataset, encoding="utf-8")))
    print(f"dataset: {dataset.name} ({len(rows)} wierszy)\n=== mapowanie ===")

    combos_by_series: dict[str, list] = defaultdict(list)
    seen_combos: set[tuple[int, int]] = set()
    unresolved: list[str] = []
    for row in rows:
        if not row.get("Model_Samochodu"):
            continue
        series = row["Seria"].strip()
        klasa = row["Klasa_Samochodu"].strip()
        car = match_entity(row["Model_Samochodu"], cars, lambda c: c["name"])
        track = match_entity(
            f"{row['Tor']} {row['Konfiguracja']}", tracks, track_label
        ) or match_entity(row["Tor"], tracks, track_label)  # config not in G61? name only
        if car is None or track is None:
            what = "auto" if car is None else "tor"
            term = row["Model_Samochodu"] if car is None else row["Tor"]
            pool = cars if car is None else tracks
            labeler = (lambda c: c["name"]) if car is None else track_label
            near = [labeler(x) for x in pool
                    if any(w in _norm(labeler(x)) for w in _norm(term).split() if len(w) > 3)][:5]
            print(f"  ?? {what}: {row['Model_Samochodu']} @ {row['Tor']} ({row['Konfiguracja']})")
            print(f"       podobne: {near}")
            unresolved.append(f"{series}: {row['Model_Samochodu']} @ {row['Tor']}")
            continue
        key = (track["id"], car["id"])
        if key in seen_combos:
            continue
        seen_combos.add(key)
        print(f"  {row['Model_Samochodu'][:28]:28s} @ {row['Tor'][:32]:32s} -> "
              f"{car['name'][:30]:30s} @ {track_label(track)}")
        combos_by_series[series].append(
            (track["id"], track_label(track), car["id"], car["name"], klasa)
        )

    if map_only:
        return 0

    total = HarvestStats()
    for series, combos in combos_by_series.items():
        slug = "".join(c if c.isalnum() else "-" for c in series.lower())[:30]
        config = HarvestConfig(
            cache_dir=Path.home() / ".iracing_analysis" / "g61_cache",
            per_combo=12,
            refs=2,
            max_laps=200,
            dry_run=dry_run,
            tags=("g61", "harvest", f"series:{slug}"),
        )
        stats = harvest(client, library, combos, config)
        print(f"\n=== {series}: listed {stats.listed}, selected {stats.selected}, "
              f"downloaded {stats.downloaded}, imported {stats.imported}, "
              f"rejected {stats.rejected}, skipped {stats.skipped}, errors {stats.errors}")
        for attr in ("listed", "selected", "downloaded", "imported", "rejected", "skipped", "errors"):
            setattr(total, attr, getattr(total, attr) + getattr(stats, attr))

    if not dry_run:
        out = Path("/Users/fifeek/PycharmProjects/Iracing/racing_real_summaries.jsonl")
        ex = export_summaries(telemetry_dirs=[], library_dirs=[library.root], out_path=out)
        print(f"\nexport: +{ex.written} summaries -> {out}")

    print(f"\nTOTAL: downloaded {total.downloaded}, imported {total.imported}, "
          f"rejected {total.rejected}, skipped {total.skipped}, errors {total.errors}")
    if unresolved:
        print("\nNIEROZWIĄZANE:")
        for u in unresolved:
            print(f"  - {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
