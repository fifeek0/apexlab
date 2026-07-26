"""Hover readout: every selected lap's values (and the spread) at the cursor.

Items are created once and updated in place — the readout refreshes on
every cursor move, so allocations here directly translate to cursor lag.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from ..analysis.workspace import Workspace
from .cursor import CursorController
from .palette import CHANNEL_FORMATS, lap_color

__all__ = ["HoverReadout"]

_COLUMNS = ("Delta", "Speed", "Throttle", "Brake", "SteeringWheelAngle", "RPM", "Gear", "Time")


class HoverReadout(QTableWidget):
    def __init__(self, workspace: Workspace, cursor: CursorController, parent=None):
        self.workspace = workspace
        self.cursor = cursor
        self._columns = [
            c
            for c in _COLUMNS
            if c in ("Delta", "Time") or c in workspace.aligned.laps[0].channels
        ]
        n_laps = len(workspace.aligned.laps)
        super().__init__(n_laps + 1, len(self._columns) + 1, parent)

        headers = ["Lap"] + [CHANNEL_FORMATS.get(c, (c, 1.0, ""))[0] for c in self._columns]
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().hide()
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # build all items once; _update only rewrites their text
        for i, al in enumerate(workspace.aligned.laps):
            label = QTableWidgetItem(al.label())
            label.setForeground(lap_color(i))
            self.setItem(i, 0, label)
            for j in range(len(self._columns)):
                self.setItem(i, j + 1, QTableWidgetItem(""))
        self.setItem(n_laps, 0, QTableWidgetItem("spread"))
        for j in range(len(self._columns)):
            self.setItem(n_laps, j + 1, QTableWidgetItem(""))

        self.cursor.distanceChanged.connect(self._update)
        self._update(cursor.distance)

    def value_at(self, row: int, channel: str) -> float:
        """Displayed (converted) numeric value — used by tests."""
        col = self._columns.index(channel) + 1
        item = self.item(row, col)
        return float(item.text().replace("+", "")) if item else float("nan")

    def _update(self, distance: float) -> None:
        sample = self.workspace.cursor_values(distance)
        n_laps = len(sample.laps)
        for i, row_values in enumerate(sample.laps):
            for j, channel in enumerate(self._columns):
                _, factor, fmt = CHANNEL_FORMATS.get(channel, (channel, 1.0, "{:.2f}"))
                self.item(i, j + 1).setText(fmt.format(row_values[channel] * factor))
        for j, channel in enumerate(self._columns):
            _, factor, fmt = CHANNEL_FORMATS.get(channel, (channel, 1.0, "{:.2f}"))
            text = fmt.format(sample.spread[channel] * factor).replace("+", "")
            self.item(n_laps, j + 1).setText(text)
