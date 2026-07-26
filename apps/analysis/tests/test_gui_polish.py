"""GUI responsiveness/quality fixes: item reuse, legend, tab persistence."""

from __future__ import annotations

import pytest


def test_readout_reuses_items_on_cursor_move(qtbot, workspace) -> None:
    """The readout must update item text in place — rebuilding items on every
    mouse move is what made the cursor laggy."""
    from iracing_analysis.gui.cursor import CursorController
    from iracing_analysis.gui.readout import HoverReadout

    cursor = CursorController(grid_end=float(workspace.grid[-1]))
    readout = HoverReadout(workspace, cursor)
    qtbot.addWidget(readout)

    speed_col = readout._columns.index("Speed") + 1
    cursor.set_distance(500.0)
    item_before = readout.item(0, speed_col)
    text_before = item_before.text()
    cursor.set_distance(1500.0)
    item_after = readout.item(0, speed_col)

    assert item_after is item_before  # same object, no reallocation
    assert item_after.text() != text_before  # but the value changed


def test_traces_have_legend_and_no_si_prefix(qtbot, workspace) -> None:
    from iracing_analysis.gui.cursor import CursorController
    from iracing_analysis.gui.traces import TraceView

    cursor = CursorController(grid_end=float(workspace.grid[-1]))
    view = TraceView(workspace, cursor)
    qtbot.addWidget(view)

    assert view.plots[0].legend is not None
    for plot in view.plots:
        assert plot.getAxis("left").autoSIPrefix is False


def test_reference_switch_keeps_current_tab(qtbot, clean_laps, tmp_path) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=str(tmp_path), library=LapLibrary(tmp_path / "lib"))
    qtbot.addWidget(win)
    win.open_laps(clean_laps)

    # user is looking at Corners, then switches the reference lap
    for i in range(win.tabs.count()):
        if win.tabs.tabText(i) == "Corners":
            win.tabs.setCurrentIndex(i)
            break
    before = win.tabs.tabText(win.tabs.currentIndex())
    win.reference_combo.setCurrentIndex(1)
    assert win.tabs.tabText(win.tabs.currentIndex()) == before
