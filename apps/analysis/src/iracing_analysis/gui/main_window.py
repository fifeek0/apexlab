"""Main window: session browser dock + analysis tabs."""

from __future__ import annotations

import logging
from pathlib import Path

from iracing_core import LapData, LapLibrary, default_telemetry_dir, scan_telemetry_dir
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QToolBar,
    QWidget,
)

from ..analysis.workspace import Workspace
from ..config import AppConfig, load_config
from .browser import SessionBrowser
from .cursor import CursorController
from .readout import HoverReadout
from .trackmap_view import TrackMapView
from .traces import TraceView

__all__ = ["MainWindow"]

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        telemetry_dir: str | Path | None = None,
        config: AppConfig | None = None,
        library: "LapLibrary | None" = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config or load_config()
        self.telemetry_dir = Path(
            telemetry_dir or self.config.telemetry_dir or default_telemetry_dir()
        )
        self.workspace: Workspace | None = None
        self.cursor: CursorController | None = None
        self.library = library or LapLibrary(
            self.config.library_dir or Path.home() / ".iracing_analysis" / "library"
        )

        self.setWindowTitle("iRacing Telemetry Analysis")
        self.resize(1500, 950)

        # -- browser dock -------------------------------------------------
        self.browser = SessionBrowser()
        self.browser.lapsSelected.connect(self.open_laps)
        dock = QDockWidget("Sessions", self)
        dock.setWidget(self.browser)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # -- toolbar -------------------------------------------------------
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction("Open folder…", self._choose_folder)
        toolbar.addAction("Rescan", self.refresh_sessions)
        toolbar.addAction("Settings…", self._open_settings)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Reference lap: "))
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(self._reference_changed)
        toolbar.addWidget(self.reference_combo)

        # -- central tabs ---------------------------------------------------
        from .library_view import LibraryView

        self.tabs = QTabWidget()
        placeholder = QLabel(
            "Scan a telemetry folder, tick laps in the browser and hit "
            "'Analyze selected laps'."
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(placeholder, "Welcome")
        self.library_view = LibraryView(self.library)
        self.library_view.addToAnalysis.connect(self.add_lap_to_analysis)
        self.tabs.addTab(self.library_view, "Library")
        self.setCentralWidget(self.tabs)

        self.status_distance = QLabel("— m")
        self.statusBar().addPermanentWidget(self.status_distance)

    # -- session browsing -----------------------------------------------------

    def refresh_sessions(self) -> None:
        groups = scan_telemetry_dir(self.telemetry_dir)
        self.browser.populate(groups)
        self.statusBar().showMessage(
            f"{len(groups)} session(s) found in {self.telemetry_dir}", 5000
        )

    def _open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        dialog = SettingsDialog(self.config, parent=self)
        if dialog.exec():
            if self.config.telemetry_dir:
                self.telemetry_dir = Path(self.config.telemetry_dir)
                self.refresh_sessions()
            if self.workspace is not None:
                self._rebuild_tabs()

    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose telemetry folder", str(self.telemetry_dir)
        )
        if chosen:
            self.telemetry_dir = Path(chosen)
            self.refresh_sessions()

    # -- analysis ---------------------------------------------------------------

    def open_laps(self, laps: list[LapData]) -> None:
        """Build a workspace from the chosen laps and (re)create all views."""
        tracks = {lap.meta.track_name for lap in laps if lap.meta}
        if len(tracks) > 1:
            self.statusBar().showMessage(
                f"Select laps from one track only (got: {', '.join(sorted(tracks))})", 8000
            )
            return

        self.workspace = Workspace(laps, spacing=self.config.grid_spacing_m)
        self.cursor = CursorController(grid_end=float(self.workspace.grid[-1]), parent=self)
        self.cursor.distanceChanged.connect(
            lambda d: self.status_distance.setText(f"{d:7.1f} m")
        )

        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.reference_combo.addItems(self.workspace.lap_labels())
        self.reference_combo.setCurrentIndex(self.workspace.reference_index)
        self.reference_combo.blockSignals(False)

        self._rebuild_tabs()

    def _rebuild_tabs(self) -> None:
        assert self.workspace is not None and self.cursor is not None
        current = self.tabs.currentIndex()
        self.tabs.clear()

        from .dashboard import AnalysisDashboard

        provider = None
        if self.config.ai.enabled:
            from ..insights import create_provider

            provider = create_provider(self.config.ai)
        self.tabs.addTab(
            AnalysisDashboard(self.workspace, provider=provider, cursor=self.cursor),
            "Pit Wall",
        )

        traces_split = QSplitter(Qt.Orientation.Vertical)
        traces_split.addWidget(TraceView(self.workspace, self.cursor))
        readout = HoverReadout(self.workspace, self.cursor)
        readout.setMaximumHeight(190)
        traces_split.addWidget(readout)
        traces_split.setStretchFactor(0, 5)
        traces_split.setStretchFactor(1, 1)
        self.tabs.addTab(traces_split, "Traces")

        self.tabs.addTab(TrackMapView(self.workspace, self.cursor), "Track Map")
        self._add_extra_tabs()
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)

    def _add_extra_tabs(self) -> None:
        """Analysis tabs beyond traces/map; extended phase by phase."""
        assert self.workspace is not None and self.cursor is not None
        from .consistency_view import ConsistencyView
        from .corners_view import CornersView
        from .gg_view import GGView
        from .sectors_view import SectorsView

        self.tabs.addTab(
            SectorsView(self.workspace, minisector_count=self.config.minisector_count),
            "Sectors",
        )
        if len(self.workspace.laps) > 1:
            self.tabs.addTab(CornersView(self.workspace, self.cursor), "Corners")
        self.tabs.addTab(GGView(self.workspace), "GG Diagram")
        self.tabs.addTab(ConsistencyView(self.workspace), "Consistency")
        self.tabs.addTab(self.library_view, "Library")
        if len(self.workspace.laps) > 1:
            from .insights_view import InsightsView

            self.tabs.addTab(InsightsView(self.workspace, self.config.ai), "AI Report")

    def add_lap_to_analysis(self, lap: LapData) -> None:
        """Merge a (library) lap into the current analysis workspace."""
        if self.workspace is None:
            self.open_laps([lap])
            return
        current_track = self.workspace.laps[0].meta.track_name if self.workspace.laps[0].meta else ""
        lap_track = lap.meta.track_name if lap.meta else ""
        if current_track and lap_track and current_track != lap_track:
            self.statusBar().showMessage(
                f"Library lap is from '{lap_track}', current analysis is '{current_track}'", 8000
            )
            return
        self.open_laps([*self.workspace.laps, lap])

    def _reference_changed(self, index: int) -> None:
        if self.workspace is None or index < 0:
            return
        self.workspace.set_reference(index)
        self._rebuild_tabs()

    def current_view(self) -> QWidget:
        return self.tabs.currentWidget()
