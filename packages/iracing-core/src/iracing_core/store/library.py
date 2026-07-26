"""Personal reference-lap library: SQLite index + parquet channel cache.

This is the free replacement for paid tools' global lap databases: build it
from your own best laps and from ``.ibt`` files shared by faster drivers /
teammates. Laps are tagged and filterable (track, car, tags, reference
flag) and load back as full :class:`~iracing_core.models.LapData`, directly
usable in alignment/delta analysis alongside freshly parsed laps.

Layout on disk::

    <root>/
      library.db          SQLite index (laps + tags)
      laps/000001.parquet one file per lap, all loaded channels
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..ibt import IbtReader
from ..models import LapData, SessionMeta
from ..sessions import extract_laps

__all__ = ["LapLibrary", "LapRecord"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS laps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_name TEXT NOT NULL,
    track_display_name TEXT,
    track_config TEXT,
    track_length_m REAL,
    car_path TEXT,
    car_screen_name TEXT,
    driver_name TEXT,
    lap_number INTEGER,
    lap_time REAL,
    crossing_start_time REAL,
    crossing_end_time REAL,
    session_date TEXT,
    imported_at TEXT NOT NULL,
    is_reference INTEGER NOT NULL DEFAULT 0,
    source_file TEXT,
    data_file TEXT NOT NULL,
    sector_starts TEXT,
    notes TEXT DEFAULT '',
    UNIQUE (source_file, lap_number)
);
CREATE TABLE IF NOT EXISTS tags (
    lap_id INTEGER NOT NULL REFERENCES laps(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    UNIQUE (lap_id, tag)
);
"""


@dataclass(frozen=True)
class LapRecord:
    """Index entry of one stored lap."""

    lap_id: int
    track_name: str
    track_display_name: str
    track_config: str
    track_length_m: float | None
    car_path: str
    car_screen_name: str
    driver_name: str
    lap_number: int
    lap_time: float | None
    session_date: str | None
    imported_at: str
    is_reference: bool
    source_file: str | None
    data_file: str
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        t = "--:--.---"
        if self.lap_time is not None:
            mins, secs = divmod(self.lap_time, 60.0)
            t = f"{int(mins)}:{secs:06.3f}"
        who = self.driver_name or "?"
        return f"{t} — {who} — {self.track_display_name} ({self.car_screen_name})"


class LapLibrary:
    """Local lap store. Thread-unsafe by design (desktop app usage)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.laps_dir = self.root / "laps"
        self.laps_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "library.db")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- writing ----------------------------------------------------------

    def add_lap(
        self,
        lap: LapData,
        tags: tuple[str, ...] = (),
        is_reference: bool = False,
        source_file: str | Path | None = None,
        notes: str = "",
    ) -> LapRecord:
        """Store one lap (channels + metadata). Re-adding the same
        ``(source_file, lap_number)`` returns the existing record."""
        meta = lap.meta or SessionMeta(path=Path(source_file or "unknown"))
        source = str(source_file or meta.path)

        existing = self._db.execute(
            "SELECT id FROM laps WHERE source_file = ? AND lap_number = ?",
            (source, lap.lap_number),
        ).fetchone()
        if existing:
            return self.get_record(int(existing[0]))

        cur = self._db.execute(
            """INSERT INTO laps (track_name, track_display_name, track_config,
                track_length_m, car_path, car_screen_name, driver_name,
                lap_number, lap_time, crossing_start_time, crossing_end_time,
                session_date, imported_at, is_reference, source_file,
                data_file, sector_starts, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meta.track_name,
                meta.track_display_name,
                meta.track_config,
                meta.track_length_m,
                meta.car_path,
                meta.car_screen_name,
                meta.driver_name,
                lap.lap_number,
                lap.lap_time,
                lap.crossing_start_time,
                lap.crossing_end_time,
                meta.session_start_date.isoformat() if meta.session_start_date else None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                int(is_reference),
                source,
                "",  # patched after we know the id
                json.dumps(meta.sector_starts_pct),
                notes,
            ),
        )
        lap_id = int(cur.lastrowid)
        data_file = f"laps/{lap_id:06d}.parquet"
        self._db.execute("UPDATE laps SET data_file = ? WHERE id = ?", (data_file, lap_id))
        for tag in tags:
            self._db.execute(
                "INSERT OR IGNORE INTO tags (lap_id, tag) VALUES (?, ?)", (lap_id, tag)
            )
        self._db.commit()

        frame = pd.DataFrame({name: np.asarray(arr) for name, arr in lap.channels.items()})
        frame.to_parquet(self.root / data_file, index=False)
        return self.get_record(lap_id)

    def import_ibt(
        self,
        path: str | Path,
        laps: str | list[int] = "best",
        tags: tuple[str, ...] = (),
        is_reference: bool = False,
    ) -> list[LapRecord]:
        """Import laps from a ``.ibt`` file (your own or a faster driver's).

        ``laps``: ``'best'`` (fastest clean lap), ``'clean'`` (all clean),
        ``'all'`` (every complete lap), or a list of lap numbers.
        """
        path = Path(path)
        with IbtReader(path) as reader:
            all_laps = extract_laps(reader)

        if laps == "best":
            candidates = [lap for lap in all_laps if lap.is_clean and lap.lap_time]
            chosen = [min(candidates, key=lambda lap: lap.lap_time)] if candidates else []
        elif laps == "clean":
            chosen = [lap for lap in all_laps if lap.is_clean]
        elif laps == "all":
            chosen = [lap for lap in all_laps if lap.is_complete]
        else:
            wanted = set(laps)
            chosen = [lap for lap in all_laps if lap.lap_number in wanted]

        return [
            self.add_lap(lap, tags=tags, is_reference=is_reference, source_file=path)
            for lap in chosen
        ]

    # -- reading ------------------------------------------------------------

    def _record_from_row(self, row: sqlite3.Row) -> LapRecord:
        tags = tuple(
            tag
            for (tag,) in self._db.execute(
                "SELECT tag FROM tags WHERE lap_id = ? ORDER BY tag", (row["id"],)
            )
        )
        return LapRecord(
            lap_id=row["id"],
            track_name=row["track_name"],
            track_display_name=row["track_display_name"] or "",
            track_config=row["track_config"] or "",
            track_length_m=row["track_length_m"],
            car_path=row["car_path"] or "",
            car_screen_name=row["car_screen_name"] or "",
            driver_name=row["driver_name"] or "",
            lap_number=row["lap_number"],
            lap_time=row["lap_time"],
            session_date=row["session_date"],
            imported_at=row["imported_at"],
            is_reference=bool(row["is_reference"]),
            source_file=row["source_file"],
            data_file=row["data_file"],
            notes=row["notes"] or "",
            tags=tags,
        )

    def has_source(self, source_file: str | Path) -> bool:
        """Has anything from this telemetry file been imported already?"""
        row = self._db.execute(
            "SELECT 1 FROM laps WHERE source_file = ? LIMIT 1", (str(Path(source_file)),)
        ).fetchone()
        return row is not None

    def get_record(self, lap_id: int) -> LapRecord:
        self._db.row_factory = sqlite3.Row
        row = self._db.execute("SELECT * FROM laps WHERE id = ?", (lap_id,)).fetchone()
        if row is None:
            raise KeyError(f"no lap with id {lap_id}")
        return self._record_from_row(row)

    def list_laps(
        self,
        track: str | None = None,
        car: str | None = None,
        tags: tuple[str, ...] = (),
        reference_only: bool = False,
    ) -> list[LapRecord]:
        self._db.row_factory = sqlite3.Row
        query = "SELECT DISTINCT laps.* FROM laps"
        clauses, params = [], []
        if tags:
            query += " JOIN tags ON tags.lap_id = laps.id"
            clauses.append(f"tags.tag IN ({','.join('?' * len(tags))})")
            params.extend(tags)
        if track:
            clauses.append("(laps.track_name = ? OR laps.track_display_name = ?)")
            params.extend([track, track])
        if car:
            clauses.append("(laps.car_path = ? OR laps.car_screen_name = ?)")
            params.extend([car, car])
        if reference_only:
            clauses.append("laps.is_reference = 1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY laps.lap_time IS NULL, laps.lap_time"
        return [self._record_from_row(r) for r in self._db.execute(query, params)]

    def get_lap(self, lap_id: int) -> LapData:
        """Load a stored lap back as a fully usable LapData."""
        rec = self.get_record(lap_id)
        frame = pd.read_parquet(self.root / rec.data_file)
        channels = {name: frame[name].to_numpy() for name in frame.columns}
        n = len(frame)

        self._db.row_factory = sqlite3.Row
        row = self._db.execute("SELECT * FROM laps WHERE id = ?", (lap_id,)).fetchone()
        meta = SessionMeta(
            path=Path(rec.source_file or rec.data_file),
            track_name=rec.track_name,
            track_display_name=rec.track_display_name,
            track_config=rec.track_config,
            track_length_m=rec.track_length_m,
            car_path=rec.car_path,
            car_screen_name=rec.car_screen_name,
            driver_name=rec.driver_name,
            sector_starts_pct=json.loads(row["sector_starts"] or "[]"),
        )
        return LapData(
            lap_number=rec.lap_number,
            start_idx=0,
            end_idx=n,
            lap_time=rec.lap_time,
            crossing_start_time=row["crossing_start_time"],
            crossing_end_time=row["crossing_end_time"],
            is_complete=rec.lap_time is not None,
            channels=channels,
            meta=meta,
        )

    # -- updates -------------------------------------------------------------

    def set_tags(self, lap_id: int, tags: tuple[str, ...]) -> None:
        self._db.execute("DELETE FROM tags WHERE lap_id = ?", (lap_id,))
        for tag in tags:
            self._db.execute(
                "INSERT OR IGNORE INTO tags (lap_id, tag) VALUES (?, ?)", (lap_id, tag)
            )
        self._db.commit()

    def set_reference(self, lap_id: int, is_reference: bool) -> None:
        self._db.execute(
            "UPDATE laps SET is_reference = ? WHERE id = ?", (int(is_reference), lap_id)
        )
        self._db.commit()

    def set_notes(self, lap_id: int, notes: str) -> None:
        self._db.execute("UPDATE laps SET notes = ? WHERE id = ?", (notes, lap_id))
        self._db.commit()

    def delete_lap(self, lap_id: int) -> None:
        rec = self.get_record(lap_id)
        self._db.execute("DELETE FROM laps WHERE id = ?", (lap_id,))
        self._db.commit()
        data_path = self.root / rec.data_file
        if data_path.exists():
            data_path.unlink()
