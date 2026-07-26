"""Stacked, distance-aligned channel traces with one linked crosshair.

MoTeC-style layout: delta on top, then Speed / Throttle / Brake / Steering /
RPM / Gear, all x-linked on distance-from-S/F. Moving the mouse over any
plot moves a single crosshair through every plot (and, via the shared
CursorController, the track map and readout table).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, QPointF, Qt

pg.setConfigOptions(antialias=True)

from ..analysis.workspace import Workspace
from .cursor import CursorController
from .palette import lap_color

__all__ = ["TraceView", "TRACE_ROWS"]

#: (row title, channel key) — 'Delta' is computed, the rest are channels
TRACE_ROWS: tuple[tuple[str, str], ...] = (
    ("Δt vs ref [s]", "Delta"),
    ("Speed [km/h]", "Speed"),
    ("Throttle [%]", "Throttle"),
    ("Brake [%]", "Brake"),
    ("Steering [°]", "SteeringWheelAngle"),
    ("RPM", "RPM"),
    ("Gear", "Gear"),
)

_DISPLAY_FACTOR = {"Speed": 3.6, "Throttle": 100.0, "Brake": 100.0, "SteeringWheelAngle": 180.0 / np.pi}


#: relative row heights: the story rows (delta, speed) get the space
_ROW_STRETCH = {"Delta": 26, "Speed": 26, "Throttle": 12, "Brake": 12,
                "SteeringWheelAngle": 13, "RPM": 12, "Gear": 8}


class TraceView(pg.GraphicsLayoutWidget):
    def __init__(
        self,
        workspace: Workspace,
        cursor: CursorController,
        rows: tuple[tuple[str, str], ...] = TRACE_ROWS,
        corners=None,
        parent=None,
    ):
        super().__init__(parent)
        self.workspace = workspace
        self.cursor = cursor
        self.plots: list[pg.PlotItem] = []
        self.cursor_lines: list[pg.InfiniteLine] = []

        grid = workspace.grid
        available = set(workspace.aligned.laps[0].channels) | {"Delta"}
        first_plot: pg.PlotItem | None = None
        for row, (title, key) in enumerate(rows):
            if key not in available:
                continue
            plot = self.addPlot(row=row, col=0)
            plot.setLabel("left", title)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setMouseEnabled(x=True, y=False)
            if first_plot is None:
                first_plot = plot
                plot.addLegend(offset=(10, 5), colCount=len(workspace.aligned.laps))
            else:
                plot.setXLink(first_plot)
                plot.hideAxis("bottom")

            for i, al in enumerate(workspace.aligned.laps):
                values = (
                    workspace.deltas[i]
                    if key == "Delta"
                    else al.channels[key].astype(float) * _DISPLAY_FACTOR.get(key, 1.0)
                )
                color = lap_color(i)
                pen = pg.mkPen(color, width=2 if i == workspace.reference_index else 1.4)
                fill = None
                if key == "Delta" and i != workspace.reference_index:
                    fill_color = pg.mkColor(color)
                    fill_color.setAlpha(45)
                    fill = fill_color
                curve = plot.plot(
                    grid, values, pen=pen, name=al.label(),
                    fillLevel=0.0 if fill is not None else None,
                    brush=fill,
                )
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
            if key == "Delta":
                plot.addLine(y=0, pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine))

            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff88", width=1))
            plot.addItem(line, ignoreBounds=True)
            self.cursor_lines.append(line)
            self.plots.append(plot)
            self.ci.layout.setRowStretchFactor(row, _ROW_STRETCH.get(key, 12))

        if corners and first_plot is not None:
            self._add_corner_ribbon(corners, len(rows))

        if self.plots:
            self.plots[0].setXRange(float(grid[0]), float(grid[-1]), padding=0.02)
            self.plots[-1].setLabel("bottom", "Distance from S/F [m]")
            self.plots[-1].showAxis("bottom")

        self._mouse_timer = QElapsedTimer()
        self._mouse_timer.start()
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.cursor.distanceChanged.connect(self._update_cursor)
        self._update_cursor(self.cursor.distance)

    # -- corner ribbon ---------------------------------------------------------

    def _add_corner_ribbon(self, corners, row: int) -> None:
        """A thin x-linked strip of corner segments coloured by time lost —
        the lap's story at a glance, aligned with the distance axis."""
        from ..analysis.corners import analyze_corners

        ribbon = self.addPlot(row=row, col=0)
        ribbon.setXLink(self.plots[0])
        ribbon.hideAxis("left")
        ribbon.hideAxis("bottom")
        ribbon.setMouseEnabled(x=False, y=False)
        ribbon.setMaximumHeight(30)
        ribbon.setYRange(0, 1, padding=0)

        compare = next(
            (i for i in range(len(self.workspace.laps)) if i != self.workspace.reference_index),
            None,
        )
        losses = {}
        if compare is not None:
            losses = {
                comp.corner.number: comp.time_lost
                for comp in analyze_corners(self.workspace, compare, corners=corners)
            }
        max_loss = max((abs(v) for v in losses.values()), default=1.0) or 1.0

        for corner in corners:
            lost = losses.get(corner.number, 0.0)
            intensity = min(abs(lost) / max_loss, 1.0)
            if lost >= 0:
                color = pg.mkColor(226, 86, 74, int(60 + 180 * intensity))
            else:
                color = pg.mkColor(63, 191, 127, int(60 + 180 * intensity))
            region = pg.LinearRegionItem(
                values=(corner.start_m, corner.end_m), movable=False,
                brush=pg.mkBrush(color), pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            ribbon.addItem(region)
            label = pg.TextItem(corner.label(), color="#c8ccd4", anchor=(0.5, 0.5))
            label.setPos((corner.start_m + corner.end_m) / 2.0, 0.5)
            ribbon.addItem(label)

        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff88", width=1))
        ribbon.addItem(line, ignoreBounds=True)
        self.cursor_lines.append(line)
        self.ci.layout.setRowStretchFactor(row, 4)

    # -- cursor plumbing --------------------------------------------------

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        # rate-limit: readout/map updates at ~40 Hz are indistinguishable
        # from per-event updates but keep the UI perfectly fluid
        if self._mouse_timer.elapsed() < 25:
            return
        self._mouse_timer.restart()
        plot = next(
            (p for p in self.plots if p.sceneBoundingRect().contains(scene_pos)),
            self.plots[0] if self.plots else None,
        )
        if plot is None:
            return
        self.cursor.set_distance(plot.vb.mapSceneToView(scene_pos).x())

    def _update_cursor(self, distance: float) -> None:
        for line in self.cursor_lines:
            line.setValue(distance)
