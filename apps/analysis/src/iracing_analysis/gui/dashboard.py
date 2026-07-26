"""The Pit Wall: single-page analysis dashboard.

Everything on one screen, pit-wall style: the track map as the central
instrument (track line coloured by where time is lost, corner labels,
glowing scrub cursor), the distance-aligned trace stack with a corner
ribbon, the at-cursor readout, ranked corners, and the engineer's report
generated automatically by the local LLM the moment the page opens.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.corners import detect_corners
from ..analysis.theoretical import theoretical_best
from ..analysis.workspace import Workspace
from ..insights.base import InsightProvider, InsightResult
from ..insights.summary import build_summary
from .corners_view import CornersView
from .cursor import CursorController
from .palette import lap_color
from .readout import HoverReadout
from .trackmap_view import TrackMapView
from .traces import TraceView

__all__ = ["AnalysisDashboard", "CoachPanel"]

log = logging.getLogger(__name__)

_DASHBOARD_ROWS = (
    ("Δt vs ref [s]", "Delta"),
    ("Speed [km/h]", "Speed"),
    ("Throttle [%]", "Throttle"),
    ("Brake [%]", "Brake"),
    ("Steering [°]", "SteeringWheelAngle"),
    ("Gear", "Gear"),
)

_CARD_STYLE = """
QFrame#card { background: #16181c; border: 1px solid #262a31; border-radius: 6px; }
QLabel#eyebrow { color: #8b909a; font-size: 10px; letter-spacing: 2px; font-weight: 600; }
QLabel#headline { color: #e8eaee; }
QLabel#dim { color: #8b909a; }
QTextEdit { background: transparent; border: none; color: #d4d7dd; }
"""


def _mono(size: int, bold: bool = False) -> QFont:
    families = QFontDatabase.families()
    for name in ("JetBrains Mono", "Menlo", "Consolas", "DejaVu Sans Mono"):
        if name in families:
            font = QFont(name, size)
            break
    else:
        font = QFont("Monospace", size)
    font.setBold(bold)
    return font


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.---"
    mins, secs = divmod(seconds, 60.0)
    return f"{int(mins)}:{secs:06.3f}"


class _CoachWorker(QThread):
    finished_with = Signal(object)

    def __init__(self, provider: InsightProvider, summary: dict, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._summary = summary

    def run(self) -> None:
        self.finished_with.emit(self._provider.generate(self._summary))


class CoachPanel(QFrame):
    """Engineer's report: auto-generated when the pit wall opens."""

    def __init__(
        self,
        workspace: Workspace,
        provider: InsightProvider | None,
        autostart: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("card")
        self.workspace = workspace
        self.provider = provider
        self._worker: _CoachWorker | None = None

        eyebrow = QLabel("ENGINEER'S REPORT")
        eyebrow.setObjectName("eyebrow")
        self.status = QLabel()
        self.status.setObjectName("dim")
        self.regen_button = QPushButton("Regenerate")
        self.regen_button.clicked.connect(self.generate_async)

        top = QHBoxLayout()
        top.addWidget(eyebrow)
        top.addStretch(1)
        top.addWidget(self.status)
        top.addWidget(self.regen_button)

        self.report = QTextEdit()
        self.report.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.addLayout(top)
        layout.addWidget(self.report)

        analysable = len(workspace.laps) > 1
        if provider is None:
            self.status.setText("AI off — enable the endpoint in Settings")
            self.regen_button.setEnabled(False)
        elif not analysable:
            self.status.setText("add a second lap to compare")
            self.regen_button.setEnabled(False)
        elif autostart:
            QTimer.singleShot(0, self.generate_async)

    # -- generation -----------------------------------------------------------

    def _lap_index(self) -> int:
        return next(
            (i for i in range(len(self.workspace.laps)) if i != self.workspace.reference_index),
            self.workspace.reference_index,
        )

    def _summary(self) -> dict:
        return build_summary(self.workspace, self._lap_index())

    def generate_sync(self) -> InsightResult:
        assert self.provider is not None
        result = self.provider.generate(self._summary())
        self._show(result)
        return result

    def generate_async(self) -> None:
        if self.provider is None or (self._worker is not None and self._worker.isRunning()):
            return
        self.status.setText("analysing the lap…")
        self.regen_button.setEnabled(False)
        self._worker = _CoachWorker(self.provider, self._summary(), parent=self)
        self._worker.finished_with.connect(self._show)
        self._worker.finished.connect(lambda: self.regen_button.setEnabled(True))
        self._worker.start()

    def _show(self, result: InsightResult) -> None:
        if result.ok:
            self.report.setMarkdown(result.text)
            self.status.setText(f"by {result.model or result.provider}")
        else:
            self.report.setPlainText(result.text)
            self.status.setText("endpoint unavailable" if result.error != "disabled" else "AI off")

    # -- test hooks -------------------------------------------------------------

    def report_text(self) -> str:
        return self.report.toPlainText()

    def status_text(self) -> str:
        return self.status.text()


class AnalysisDashboard(QWidget):
    """Everything a driver needs on one page."""

    def __init__(
        self,
        workspace: Workspace,
        provider: InsightProvider | None = None,
        cursor: CursorController | None = None,
        autostart_coach: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.workspace = workspace
        self.cursor = cursor or CursorController(grid_end=float(workspace.grid[-1]), parent=self)
        self.setStyleSheet(_CARD_STYLE)

        corners = detect_corners(workspace.reference)

        # -- header ---------------------------------------------------------
        meta = workspace.laps[0].meta
        track = (meta.track_display_name or meta.track_name) if meta else ""
        car = meta.car_screen_name if meta else ""
        self._header_label = QLabel(f"{track}   ·   {car}".strip())
        self._header_label.setObjectName("headline")
        self._header_label.setFont(_mono(13, bold=True))

        laps_html = "   ".join(
            f"<span style='color:{lap_color(i).name()}'>{'★ ' if i == workspace.reference_index else ''}"
            f"{(al.lap.meta.driver_name + ' ') if al.lap.meta and al.lap.meta.driver_name else ''}"
            f"{_fmt_time(al.lap.lap_time)}</span>"
            for i, al in enumerate(workspace.aligned.laps)
        )
        laps_label = QLabel(laps_html)
        laps_label.setFont(_mono(12))

        delta_label = QLabel()
        delta_label.setFont(_mono(20, bold=True))
        compare = next(
            (i for i in range(len(workspace.laps)) if i != workspace.reference_index), None
        )
        if compare is not None and workspace.laps[compare].lap_time and workspace.reference.lap.lap_time:
            gap = workspace.laps[compare].lap_time - workspace.reference.lap.lap_time
            delta_label.setText(f"Δ {gap:+.3f}s")
            delta_label.setStyleSheet(
                f"color: {'#e2564a' if gap > 0 else '#3fbf7f'};"
            )
        try:
            tb = theoretical_best(workspace)
            theo_label = QLabel(f"theoretical {tb.label()}")
        except ValueError:
            theo_label = QLabel("")
        theo_label.setObjectName("dim")
        theo_label.setFont(_mono(11))

        header = QHBoxLayout()
        header.addWidget(self._header_label)
        header.addStretch(1)
        header.addWidget(laps_label)
        header.addSpacing(18)
        header.addWidget(theo_label)
        header.addSpacing(18)
        header.addWidget(delta_label)

        # -- instruments ---------------------------------------------------
        self.trackmap = TrackMapView(
            self.workspace, self.cursor, corners=corners, show_mode_combo=False
        )
        self.trackmap.set_color_mode("delta")
        self.readout = HoverReadout(self.workspace, self.cursor)
        self.readout.setMaximumHeight(132)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.trackmap, stretch=4)
        left.addWidget(self.readout, stretch=0)
        left_box = QWidget()
        left_box.setLayout(left)

        self.traces = TraceView(
            self.workspace, self.cursor, rows=_DASHBOARD_ROWS, corners=corners
        )

        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.addWidget(left_box)
        middle.addWidget(self.traces)
        middle.setStretchFactor(0, 2)
        middle.setStretchFactor(1, 3)

        # -- bottom: corners + coach -----------------------------------------
        self.corners = CornersView(self.workspace, self.cursor)
        self.corners.setMaximumHeight(230)
        self.coach = CoachPanel(self.workspace, provider, autostart=autostart_coach)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.corners)
        bottom.addWidget(self.coach)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        bottom.setMaximumHeight(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addLayout(header)
        layout.addWidget(middle, stretch=5)
        layout.addWidget(bottom, stretch=2)

    def header_text(self) -> str:
        return self._header_label.text()
