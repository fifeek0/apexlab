"""Sector / mini-sector table with per-sector best highlighting."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.sectors import (
    minisector_boundaries,
    official_sector_boundaries,
    sector_times,
)
from ..analysis.workspace import Workspace
from .palette import lap_color

__all__ = ["SectorsView"]

_BEST_BG = QColor("#1f7a33")


class SectorsView(QWidget):
    def __init__(self, workspace: Workspace, minisector_count: int = 20, parent=None):
        super().__init__(parent)
        self.workspace = workspace

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Official sectors", "Mini-sectors"])
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 200)
        self.count_spin.setValue(minisector_count)
        self.mode_combo.currentIndexChanged.connect(self._rebuild)
        self.count_spin.valueChanged.connect(self._rebuild)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("View:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("Mini-sector count:"))
        controls.addWidget(self.count_spin)
        controls.addStretch(1)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.theoretical_label = QLabel()
        self.theoretical_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table)
        layout.addWidget(self.theoretical_label)
        self._rebuild()

    def _boundaries(self) -> np.ndarray:
        if self.mode_combo.currentIndex() == 0:
            return official_sector_boundaries(self.workspace)
        return minisector_boundaries(self.workspace, self.count_spin.value())

    def _rebuild(self) -> None:
        result = sector_times(self.workspace, self._boundaries())
        n_laps, n_sectors = result.times.shape

        self.table.clear()
        self.table.setRowCount(n_laps)
        self.table.setColumnCount(n_sectors + 2)
        headers = ["Lap"] + [f"S{k + 1}" for k in range(n_sectors)] + ["Lap time"]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().hide()

        best = result.best_lap
        for i in range(n_laps):
            label_item = QTableWidgetItem(result.lap_labels[i])
            label_item.setForeground(lap_color(i))
            self.table.setItem(i, 0, label_item)
            for k in range(n_sectors):
                item = QTableWidgetItem(f"{result.times[i, k]:.3f}")
                if int(best[k]) == i:
                    item.setBackground(QBrush(_BEST_BG))
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(i, k + 1, item)
            lap_time = self.workspace.aligned.laps[i].lap.lap_time
            self.table.setItem(
                i, n_sectors + 1, QTableWidgetItem("" if lap_time is None else f"{lap_time:.3f}")
            )
        self.table.resizeColumnsToContents()
        self._update_theoretical()

    def _update_theoretical(self) -> None:
        from ..analysis.theoretical import theoretical_best

        try:
            tb = theoretical_best(self.workspace, self.count_spin.value())
        except ValueError:
            self.theoretical_label.setText("Theoretical best: n/a (no complete laps)")
            return
        self.theoretical_label.setText(
            f"<b>Theoretical best: {tb.label()}</b> — {tb.gap:.3f} s under your actual "
            f"best (from best of {tb.minisector_count} mini-sectors across the selected "
            f"laps). This is an <i>optimistic</i> upper bound: it can combine gains that "
            f"were not simultaneously sustainable."
        )

    def is_best_cell(self, row: int, column: int) -> bool:
        item = self.table.item(row, column)
        return item is not None and item.background().color() == _BEST_BG
