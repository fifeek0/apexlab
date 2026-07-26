"""Consistency tab: lap-time distribution, channel histograms, variability."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.consistency import (
    channel_histograms,
    cross_lap_variability,
    lap_time_stats,
)
from ..analysis.workspace import Workspace
from .palette import lap_color

__all__ = ["ConsistencyView"]

_HIST_CHANNELS = ("Throttle", "Brake", "Speed", "SteeringWheelAngle")


class ConsistencyView(QWidget):
    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace

        # --- headline metrics table ------------------------------------
        stats = lap_time_stats(workspace.laps)
        variability = cross_lap_variability(workspace).summary()
        rows = [
            ("Laps", f"{stats.get('count', 0)}"),
            ("Best lap", f"{stats.get('best', float('nan')):.3f} s"),
            ("Mean lap", f"{stats.get('mean', float('nan')):.3f} s"),
            ("Median lap", f"{stats.get('median', float('nan')):.3f} s"),
            ("Lap-time σ", f"{stats.get('std', float('nan')):.3f} s"),
            ("Speed σ across laps", f"{variability['speed_std_mean_ms'] * 3.6:.2f} km/h"),
            ("Throttle σ across laps", f"{variability['throttle_std_mean'] * 100:.1f} %"),
            ("Brake σ across laps", f"{variability['brake_std_mean'] * 100:.1f} %"),
        ]
        self.metrics = QTableWidget(len(rows), 2)
        self.metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics.verticalHeader().hide()
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for r, (name, value) in enumerate(rows):
            self.metrics.setItem(r, 0, QTableWidgetItem(name))
            self.metrics.setItem(r, 1, QTableWidgetItem(value))
        self.metrics.resizeColumnsToContents()
        self.metrics.setMaximumHeight(300)

        # --- lap-time distribution ---------------------------------------
        self.laptime_plot = pg.PlotWidget(title="Lap times")
        self.laptime_plot.setLabel("left", "Lap time [s]")
        self.laptime_plot.setLabel("bottom", "Lap")
        times = [lap.lap_time or np.nan for lap in workspace.laps]
        xs = np.arange(len(times))
        bars = pg.BarGraphItem(x=xs, height=times, width=0.6, brushes=[lap_color(i) for i in xs])
        self.laptime_plot.addItem(bars)
        if stats.get("best"):
            self.laptime_plot.addLine(y=stats["best"], pen=pg.mkPen("#00c853"))

        # --- channel histogram --------------------------------------------
        self.channel_combo = QComboBox()
        available = [c for c in _HIST_CHANNELS if c in workspace.aligned.laps[0].channels]
        self.channel_combo.addItems(available)
        self.channel_combo.currentTextChanged.connect(self._rebuild_histogram)
        self.hist_plot = pg.PlotWidget(title="Channel histogram (time-at-value)")
        self.hist_plot.addLegend()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Histogram channel:"))
        controls.addWidget(self.channel_combo)
        controls.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(self.metrics, stretch=1)
        top.addWidget(self.laptime_plot, stretch=2)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(self.hist_plot)
        if available:
            self._rebuild_histogram(available[0])

    def _rebuild_histogram(self, channel: str) -> None:
        self.hist_plot.clear()
        hist = channel_histograms(self.workspace, channel, bins=25)
        centers = (hist.edges[:-1] + hist.edges[1:]) / 2.0
        for i in range(hist.counts.shape[0]):
            self.hist_plot.plot(
                centers,
                hist.counts[i],
                pen=pg.mkPen(lap_color(i), width=2),
                name=hist.lap_labels[i],
            )
        self.hist_plot.setLabel("bottom", channel)
        self.hist_plot.setLabel("left", "samples")
