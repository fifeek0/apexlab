"""GG / friction-circle scatter, one colour per lap, with g-circles."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..analysis.gg import gg_points
from ..analysis.workspace import Workspace
from .palette import lap_color

__all__ = ["GGView"]


class GGView(pg.PlotWidget):
    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.setAspectLocked(True)
        self.setLabel("bottom", "Lateral acceleration [g]")
        self.setLabel("left", "Longitudinal acceleration [g]  (+accel / −brake)")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.addLegend()

        # reference circles at 1g and 2g
        theta = np.linspace(0, 2 * np.pi, 181)
        for r in (1.0, 2.0):
            self.plot(
                r * np.cos(theta),
                r * np.sin(theta),
                pen=pg.mkPen("#555555", style=pg.QtCore.Qt.PenStyle.DashLine),
            )

        self.point_items: list[pg.ScatterPlotItem] = []
        for i, al in enumerate(workspace.aligned.laps):
            gg = gg_points(al)
            color = lap_color(i)
            color.setAlpha(110)
            item = pg.ScatterPlotItem(
                x=gg.lat_g,
                y=gg.long_g,
                size=4,
                pen=None,
                brush=pg.mkBrush(color),
                name=al.label(),
            )
            self.addItem(item)
            self.point_items.append(item)
