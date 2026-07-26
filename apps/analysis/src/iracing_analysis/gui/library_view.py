"""Reference-library tab: browse/tag stored laps, import .ibt, feed analysis."""

from __future__ import annotations

import logging
from pathlib import Path

from iracing_core import LapData, LapLibrary, LapRecord
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["LibraryView"]

log = logging.getLogger(__name__)

_HEADERS = ("Time", "Driver", "Track", "Car", "Date", "Ref", "Tags", "Source")


class LibraryView(QWidget):
    """Table over :class:`iracing_core.LapLibrary` with import/tag actions."""

    addToAnalysis = Signal(object)  # LapData

    def __init__(self, library: LapLibrary, parent=None):
        super().__init__(parent)
        self.library = library
        self._records: list[LapRecord] = []

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by track / car / driver / tag…")
        self.filter_edit.textChanged.connect(self.refresh)

        import_btn = QPushButton("Import .ibt…")
        import_btn.clicked.connect(self._import_dialog)
        tag_btn = QPushButton("Edit tags…")
        tag_btn.clicked.connect(self._edit_tags)
        ref_btn = QPushButton("Toggle reference")
        ref_btn.clicked.connect(self._toggle_reference)
        add_btn = QPushButton("Add to analysis")
        add_btn.clicked.connect(self._emit_selected)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected)

        controls = QHBoxLayout()
        controls.addWidget(self.filter_edit, stretch=1)
        for b in (import_btn, tag_btn, ref_btn, add_btn, delete_btn):
            controls.addWidget(b)

        self.table = QTableWidget()
        self.table.setColumnCount(len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table)
        self.refresh()

    # -- data -----------------------------------------------------------------

    def refresh(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        records = self.library.list_laps()
        if needle:
            records = [
                r
                for r in records
                if needle
                in " ".join(
                    [r.track_name, r.track_display_name, r.car_screen_name,
                     r.car_path, r.driver_name, " ".join(r.tags)]
                ).lower()
            ]
        self._records = records
        self.table.setRowCount(len(records))
        for row, rec in enumerate(records):
            t = "" if rec.lap_time is None else f"{rec.lap_time:.3f}"
            cells = (
                t,
                rec.driver_name,
                rec.track_display_name or rec.track_name,
                rec.car_screen_name,
                (rec.session_date or "")[:10],
                "★" if rec.is_reference else "",
                ", ".join(rec.tags),
                Path(rec.source_file).name if rec.source_file else "",
            )
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _selected_record(self) -> LapRecord | None:
        row = self.table.currentRow()
        return self._records[row] if 0 <= row < len(self._records) else None

    # -- actions -------------------------------------------------------------

    def _import_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import telemetry",
            "",
            "Telemetry (*.ibt *.csv);;iRacing telemetry (*.ibt);;Garage 61 export (*.csv)",
        )
        self.import_paths(paths)

    def import_paths(self, paths: list[str]) -> None:
        """Import .ibt files (clean laps) and Garage 61 .csv exports."""
        from iracing_core import import_garage61_csv

        for path in paths:
            try:
                if str(path).lower().endswith(".csv"):
                    import_garage61_csv(self.library, path, is_reference=True)
                else:
                    self.library.import_ibt(path, laps="clean")
            except Exception as exc:
                log.warning("import of %s failed: %s", path, exc)
        if paths:
            self.refresh()

    def _edit_tags(self) -> None:
        rec = self._selected_record()
        if rec is None:
            return
        text, ok = QInputDialog.getText(
            self, "Edit tags", "Comma-separated tags:", text=", ".join(rec.tags)
        )
        if ok:
            tags = tuple(t.strip() for t in text.split(",") if t.strip())
            self.library.set_tags(rec.lap_id, tags)
            self.refresh()

    def _toggle_reference(self) -> None:
        rec = self._selected_record()
        if rec is not None:
            self.library.set_reference(rec.lap_id, not rec.is_reference)
            self.refresh()

    def _delete_selected(self) -> None:
        rec = self._selected_record()
        if rec is not None:
            self.library.delete_lap(rec.lap_id)
            self.refresh()

    def _emit_selected(self) -> None:
        rec = self._selected_record()
        if rec is not None:
            self.addToAnalysis.emit(self.library.get_lap(rec.lap_id))
