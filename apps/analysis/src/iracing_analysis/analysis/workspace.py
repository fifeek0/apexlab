"""The analysis workspace: selected laps, alignment, reference and deltas.

UI-agnostic on purpose — both the PySide6 desktop views and any alternative
front end (e.g. a local web dashboard) consume this object rather than
duplicating analysis logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from iracing_core import AlignedLap, AlignedLapSet, LapData, align_laps, delta_time

from .trackmap import shape_from_aligned_lap

__all__ = ["CursorSample", "Workspace"]


@dataclass
class CursorSample:
    """Every lap's channel values (+ delta) at one cursor distance."""

    distance: float
    index: int
    laps: list[dict[str, float]]
    spread: dict[str, float] = field(default_factory=dict)


class Workspace:
    """Selected laps of one track, aligned on a common distance grid."""

    def __init__(self, laps: list[LapData], reference_index: int = 0, spacing: float = 1.0):
        if not laps:
            raise ValueError("workspace needs at least one lap")
        self.laps = laps
        self.spacing = spacing
        self._aligned: AlignedLapSet | None = None
        self._deltas: list[np.ndarray] | None = None
        self._track_shape: tuple[np.ndarray, np.ndarray] | None = None
        self.reference_index = reference_index

    # -- alignment ---------------------------------------------------------

    @property
    def aligned(self) -> AlignedLapSet:
        if self._aligned is None:
            self._aligned = align_laps(self.laps, spacing=self.spacing)
        return self._aligned

    @property
    def grid(self) -> np.ndarray:
        return self.aligned.grid

    @property
    def reference(self) -> AlignedLap:
        return self.aligned.laps[self.reference_index]

    def set_reference(self, index: int) -> None:
        if not 0 <= index < len(self.laps):
            raise IndexError(f"reference index {index} out of range")
        self.reference_index = index
        self._deltas = None

    # -- deltas ------------------------------------------------------------

    @property
    def deltas(self) -> list[np.ndarray]:
        """Per-lap cumulative delta vs the reference (positive = slower)."""
        if self._deltas is None:
            self._deltas = [delta_time(al, self.reference) for al in self.aligned.laps]
        return self._deltas

    # -- cursor ------------------------------------------------------------

    def index_at(self, distance: float) -> int:
        return int(np.clip(np.searchsorted(self.grid, distance), 0, len(self.grid) - 1))

    def cursor_values(self, distance: float) -> CursorSample:
        idx = self.index_at(distance)
        rows: list[dict[str, float]] = []
        for i, al in enumerate(self.aligned.laps):
            row = {name: float(values[idx]) for name, values in al.channels.items()}
            row["Delta"] = float(self.deltas[i][idx])
            row["Time"] = float(al.time_at[idx])
            rows.append(row)
        spread = {
            name: max(row[name] for row in rows) - min(row[name] for row in rows)
            for name in rows[0]
        }
        return CursorSample(distance=distance, index=idx, laps=rows, spread=spread)

    # -- track geometry ------------------------------------------------------

    def track_shape(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Track outline from the reference lap, in local metres."""
        if self._track_shape is None:
            self._track_shape = shape_from_aligned_lap(
                self.reference.channels, self.reference.time_at
            )
        return self._track_shape

    def lap_lines(self) -> list[tuple[np.ndarray, np.ndarray] | None]:
        """Every lap's GPS driving line, projected into ONE local frame
        (the reference lap's centroid) so the lines can be compared on the
        map — zoom into a corner and the driven lines separate."""
        from .trackmap import latlon_to_local_xy

        ref = self.reference.channels
        if "Lat" not in ref or "Lon" not in ref:
            return [None] * len(self.laps)
        origin = (float(ref["Lat"].mean()), float(ref["Lon"].mean()))
        lines: list[tuple[np.ndarray, np.ndarray] | None] = []
        for al in self.aligned.laps:
            if "Lat" in al.channels and "Lon" in al.channels:
                lines.append(latlon_to_local_xy(al.channels["Lat"], al.channels["Lon"], origin))
            else:
                lines.append(None)
        return lines

    # -- misc ----------------------------------------------------------------

    def lap_labels(self) -> list[str]:
        return [al.label() for al in self.aligned.laps]

    @property
    def meta(self):
        return self.laps[0].meta
