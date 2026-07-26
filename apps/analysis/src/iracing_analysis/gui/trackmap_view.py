"""Track map synced to the linked cursor.

Outline + cursor marker + click-to-seek, with optional colouring of the
track path by reference-lap speed or by delta vs the reference (green =
gaining, red = losing).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QGraphicsProxyWidget

from ..analysis.workspace import Workspace
from .cursor import CursorController

__all__ = ["TrackMapView"]

_SPEED_CMAP = pg.ColorMap(
    np.linspace(0.0, 1.0, 5),
    [(60, 60, 220, 255), (0, 180, 255, 255), (0, 220, 120, 255), (255, 220, 0, 255), (255, 60, 60, 255)],
)
_DELTA_CMAP = pg.ColorMap(
    np.linspace(0.0, 1.0, 3),
    [(0, 220, 120, 255), (150, 150, 150, 255), (255, 60, 60, 255)],
)


class TrackMapView(pg.PlotWidget):
    def __init__(
        self,
        workspace: Workspace,
        cursor: CursorController,
        compare_index: int | None = None,
        corners=None,
        show_mode_combo: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.workspace = workspace
        self.cursor = cursor
        self.color_mode: str | None = None
        # lap whose delta colours the map: first non-reference lap by default
        if compare_index is None:
            compare_index = next(
                (i for i in range(len(workspace.laps)) if i != workspace.reference_index),
                workspace.reference_index,
            )
        self.compare_index = compare_index
        self._corner_labels: list[str] = []
        self._show_mode_combo = show_mode_combo

        self.setAspectLocked(True)
        self.hideAxis("left")
        self.hideAxis("bottom")

        self._shape = workspace.track_shape()
        self._points: pg.ScatterPlotItem | None = None
        if self._shape is not None:
            x, y = self._shape
            self._points = pg.ScatterPlotItem(
                x=x, y=y, size=8, pen=None, brush=pg.mkBrush("#5b5f68"), pxMode=True
            )
            self.addItem(self._points)
            self.addItem(
                pg.ScatterPlotItem(
                    [float(x[0])], [float(y[0])], symbol="s", size=12,
                    brush="#e8eaee", pen=pg.mkPen("#101114", width=1),
                )
            )
            if corners:
                self._add_corner_labels(corners)

        # per-lap GPS driving lines in one shared frame: zoom into a corner
        # (mouse wheel) and the driven lines separate, Garage-61 style
        self.lap_line_items: list[pg.PlotCurveItem] = []
        from .palette import lap_color

        for i, line in enumerate(workspace.lap_lines()):
            if line is None:
                continue
            color = pg.mkColor(lap_color(i))
            color.setAlpha(210)
            item = pg.PlotCurveItem(
                line[0], line[1],
                pen=pg.mkPen(color, width=2 if i == workspace.reference_index else 1.6),
            )
            item.setZValue(5)
            self.addItem(item)
            self.lap_line_items.append(item)
        # scrub cursor: soft halo + hard core, so it reads on any track colour
        self._halo = pg.ScatterPlotItem(size=30, brush=pg.mkBrush(230, 159, 0, 70), pen=None)
        self._marker = pg.ScatterPlotItem(
            size=13, brush="#e69f00", pen=pg.mkPen("#101114", width=2)
        )
        self.addItem(self._halo)
        self.addItem(self._marker)

        # colour-mode selector rendered inside the plot corner
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Plain", "Colour: speed", "Colour: delta"])
        self.mode_combo.currentIndexChanged.connect(self._mode_combo_changed)
        if self._show_mode_combo:
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(self.mode_combo)
            self.plotItem.layout.addItem(proxy, 4, 1)

        self.cursor.distanceChanged.connect(self._update_cursor)
        self.scene().sigMouseClicked.connect(self._on_click)
        self._update_cursor(self.cursor.distance)

    # -- corner labels ---------------------------------------------------------

    def _add_corner_labels(self, corners) -> None:
        """Place T1..Tn at the apexes, nudged outward from the track centroid."""
        x, y = self._shape
        cx, cy = float(np.mean(x)), float(np.mean(y))
        for corner in corners:
            idx = self.workspace.index_at(corner.apex_m)
            px, py = float(x[idx]), float(y[idx])
            dx, dy = px - cx, py - cy
            norm = max((dx**2 + dy**2) ** 0.5, 1e-9)
            offset = 55.0
            label = pg.TextItem(corner.label(), color="#9aa0ab", anchor=(0.5, 0.5))
            label.setPos(px + dx / norm * offset, py + dy / norm * offset)
            self.addItem(label)
            self._corner_labels.append(corner.label())

    def corner_labels(self) -> list[str]:
        return list(self._corner_labels)

    # -- colouring -----------------------------------------------------------

    def _mode_combo_changed(self, index: int) -> None:
        self.set_color_mode({0: None, 1: "speed", 2: "delta"}[index])

    def set_color_mode(self, mode: str | None) -> None:
        """``None`` (plain), ``'speed'`` or ``'delta'``."""
        if mode not in (None, "speed", "delta"):
            raise ValueError(f"unknown colour mode {mode!r}")
        self.color_mode = mode
        if self._points is None:
            return
        n = len(self.workspace.grid)
        if mode is None:
            self._points.setBrush(pg.mkBrush("#666666"))
        elif mode == "speed":
            speed = self.workspace.reference.channels["Speed"].astype(float)
            norm = (speed - speed.min()) / max(float(np.ptp(speed)), 1e-9)
            self._points.setBrush([pg.mkBrush(c) for c in _SPEED_CMAP.mapToQColor(norm)])
        else:
            delta = self.workspace.deltas[self.compare_index]
            local = np.gradient(delta, self.workspace.grid)  # s lost per metre
            limit = max(float(np.abs(local).max()), 1e-6)
            norm = 0.5 + 0.5 * np.clip(local / limit, -1.0, 1.0)
            self._points.setBrush([pg.mkBrush(c) for c in _DELTA_CMAP.mapToQColor(norm)])
        assert self._points.data.shape[0] == n

    # -- cursor plumbing --------------------------------------------------------

    def marker_pos(self) -> tuple[float, float]:
        x, y = self._marker.getData()
        return (float(x[0]), float(y[0])) if len(x) else (float("nan"), float("nan"))

    def _update_cursor(self, distance: float) -> None:
        if self._shape is None:
            return
        idx = self.workspace.index_at(distance)
        x, y = self._shape
        self._marker.setData([float(x[idx])], [float(y[idx])])
        self._halo.setData([float(x[idx])], [float(y[idx])])

    def _on_click(self, event) -> None:
        if self._shape is None:
            return
        pos = self.plotItem.vb.mapSceneToView(event.scenePos())
        x, y = self._shape
        idx = int(np.argmin((x - pos.x()) ** 2 + (y - pos.y()) ** 2))
        self.cursor.set_distance(float(self.workspace.grid[idx]))
