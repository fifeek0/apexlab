"""Phase 4 GUI: sector table with best highlights + ranked corner view."""

from __future__ import annotations

import pytest


def test_sectors_view_official_and_minisectors(qtbot, workspace) -> None:
    from iracing_analysis.analysis.sectors import official_sector_boundaries, sector_times
    from iracing_analysis.gui.sectors_view import SectorsView

    view = SectorsView(workspace, minisector_count=12)
    qtbot.addWidget(view)

    table = sector_times(workspace, official_sector_boundaries(workspace))
    assert view.table.rowCount() == len(workspace.laps)
    assert view.table.columnCount() == table.n_sectors + 2  # lap label + lap time

    # values match the analysis lib
    for i in range(len(workspace.laps)):
        for k in range(table.n_sectors):
            cell = float(view.table.item(i, k + 1).text())
            assert cell == pytest.approx(table.times[i, k], abs=5e-4)

    # best lap per sector is highlighted
    for k, best in enumerate(table.best_lap):
        assert view.is_best_cell(int(best), k + 1)

    # switching to mini-sectors rebuilds the table
    view.mode_combo.setCurrentIndex(1)
    assert view.table.columnCount() == 12 + 2


def test_corners_view_ranked_and_cursor_jump(qtbot, workspace) -> None:
    from iracing_analysis.analysis.corners import analyze_corners, rank_by_time_lost
    from iracing_analysis.gui.corners_view import CornersView
    from iracing_analysis.gui.cursor import CursorController

    cursor = CursorController(grid_end=float(workspace.grid[-1]))
    view = CornersView(workspace, cursor)
    qtbot.addWidget(view)

    lap_index = view.current_lap_index()
    ranked = rank_by_time_lost(analyze_corners(workspace, lap_index))
    assert view.table.rowCount() == len(ranked)

    # first row is the worst corner; clicking it moves the shared cursor
    view.jump_to_row(0)
    assert cursor.distance == pytest.approx(ranked[0].corner.apex_m)


def test_main_window_has_phase4_tabs(qtbot, clean_laps) -> None:
    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=".")
    qtbot.addWidget(win)
    win.open_laps(clean_laps)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Sectors" in tabs and "Corners" in tabs
