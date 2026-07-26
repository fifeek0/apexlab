"""Single source of truth for the linked cursor position (track distance)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

__all__ = ["CursorController"]


class CursorController(QObject):
    """Holds the cursor's distance-around-the-lap; every view follows it."""

    distanceChanged = Signal(float)

    def __init__(self, grid_end: float, parent: QObject | None = None):
        super().__init__(parent)
        self._distance = 0.0
        self._grid_end = grid_end

    @property
    def distance(self) -> float:
        return self._distance

    def set_distance(self, distance: float) -> None:
        clamped = min(max(0.0, float(distance)), self._grid_end)
        if clamped != self._distance:
            self._distance = clamped
            self.distanceChanged.emit(clamped)
