"""Session/lap browser: telemetry folder → sessions → laps to analyse."""

from __future__ import annotations

import logging
from pathlib import Path

from iracing_core import IbtReader, LapData, SessionGroup, extract_laps
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = ["SessionBrowser"]

log = logging.getLogger(__name__)

_ROLE_PATH = Qt.ItemDataRole.UserRole
_ROLE_LAP = Qt.ItemDataRole.UserRole + 1


class SessionBrowser(QWidget):
    """Tree of sessions → files → laps, with checkable laps.

    Laps of a file are parsed lazily on first expansion; checked laps are
    emitted through :attr:`lapsSelected` when the user hits *Analyze*.
    """

    lapsSelected = Signal(list)  # list[LapData]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Session / Lap", "Time", "Flags"])
        self.tree.itemExpanded.connect(self._on_expanded)
        self._lap_cache: dict[Path, list[LapData]] = {}

        self.analyze_button = QPushButton("Analyze selected laps")
        self.analyze_button.clicked.connect(self._emit_selection)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        layout.addWidget(self.analyze_button)

    # -- population --------------------------------------------------------

    def populate(self, groups: list[SessionGroup]) -> None:
        self.tree.clear()
        self._lap_cache.clear()
        for group in groups:
            top = QTreeWidgetItem([group.label(), "", ""])
            self.tree.addTopLevelItem(top)
            for meta in group.files:
                file_item = QTreeWidgetItem([meta.path.name, "", ""])
                file_item.setData(0, _ROLE_PATH, str(meta.path))
                file_item.setChildIndicatorPolicy(
                    QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
                )
                top.addChild(file_item)
            top.setExpanded(True)

    def session_count(self) -> int:
        return self.tree.topLevelItemCount()

    # -- lazy lap loading -----------------------------------------------------

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        path_str = item.data(0, _ROLE_PATH)
        if not path_str or item.childCount() > 0:
            return
        self.load_laps_into(item)

    def load_laps_into(self, file_item: QTreeWidgetItem) -> list[LapData]:
        path = Path(file_item.data(0, _ROLE_PATH))
        if path not in self._lap_cache:
            try:
                with IbtReader(path) as reader:
                    self._lap_cache[path] = extract_laps(reader)
            except Exception as exc:
                log.warning("cannot parse %s: %s", path, exc)
                self._lap_cache[path] = []
        for lap in self._lap_cache[path]:
            flags = []
            if lap.is_out_lap:
                flags.append("out")
            if lap.is_in_lap:
                flags.append("in")
            if lap.touched_pits and not (lap.is_out_lap or lap.is_in_lap):
                flags.append("pit")
            if not lap.is_complete:
                flags.append("partial")
            label = f"Lap {lap.lap_number}"
            time_txt = lap.label().split(" ", 1)[1]
            lap_item = QTreeWidgetItem([label, time_txt, ",".join(flags) or "clean"])
            lap_item.setData(0, _ROLE_LAP, lap)
            lap_item.setFlags(lap_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            lap_item.setCheckState(0, Qt.CheckState.Unchecked)
            file_item.addChild(lap_item)
        file_item.setExpanded(True)
        return self._lap_cache[path]

    # -- selection -------------------------------------------------------------

    def checked_laps(self) -> list[LapData]:
        laps: list[LapData] = []

        def walk(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                lap = child.data(0, _ROLE_LAP)
                if lap is not None and child.checkState(0) == Qt.CheckState.Checked:
                    laps.append(lap)
                walk(child)

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        return laps

    def _emit_selection(self) -> None:
        laps = self.checked_laps()
        if laps:
            self.lapsSelected.emit(laps)
