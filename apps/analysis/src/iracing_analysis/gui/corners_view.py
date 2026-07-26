"""Ranked per-corner time-loss view with click-to-jump cursor sync."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.corners import CornerComparison, analyze_corners, rank_by_time_lost
from ..analysis.workspace import Workspace
from .cursor import CursorController

__all__ = ["CornersView"]

_HEADERS = (
    "Corner",
    "Apex [m]",
    "Time lost [s]",
    "Entry [s]",
    "Exit [s]",
    "Next straight [s]",
    "Min speed [km/h]",
    "Ref min [km/h]",
    "Likely why",
)


class CornersView(QWidget):
    """Corners ranked by total time lost vs the reference lap."""

    def __init__(self, workspace: Workspace, cursor: CursorController, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.cursor = cursor
        self._ranked: list[CornerComparison] = []

        self.lap_combo = QComboBox()
        for i, label in enumerate(workspace.lap_labels()):
            if i != workspace.reference_index:
                self.lap_combo.addItem(f"{label}", userData=i)
        self.lap_combo.currentIndexChanged.connect(self._rebuild)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Analyse lap:"))
        controls.addWidget(self.lap_combo)
        controls.addWidget(
            QLabel(f" vs reference {workspace.lap_labels()[workspace.reference_index]}")
        )
        controls.addStretch(1)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(lambda row, _col: self.jump_to_row(row))

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table)
        self._rebuild()

    def current_lap_index(self) -> int:
        data = self.lap_combo.currentData()
        return int(data) if data is not None else self.workspace.reference_index

    def jump_to_row(self, row: int) -> None:
        if 0 <= row < len(self._ranked):
            self.cursor.set_distance(self._ranked[row].corner.apex_m)

    def _rebuild(self) -> None:
        lap_index = self.current_lap_index()
        if lap_index == self.workspace.reference_index:
            self._ranked = []
        else:
            self._ranked = rank_by_time_lost(analyze_corners(self.workspace, lap_index))

        self.table.clear()
        self.table.setRowCount(len(self._ranked))
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().hide()

        for row, comp in enumerate(self._ranked):
            cells = (
                comp.corner.label(),
                f"{comp.corner.apex_m:.0f}",
                f"{comp.time_lost:+.3f}",
                f"{comp.delta_entry:+.3f}",
                f"{comp.delta_exit:+.3f}",
                f"{comp.delta_carry:+.3f}",
                f"{comp.metrics.min_speed * 3.6:.1f}",
                f"{comp.ref_metrics.min_speed * 3.6:.1f}",
                "; ".join(comp.reasons),
            )
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
