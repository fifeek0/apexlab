"""Phase 3 gate: linked cursor identical across all trace plots + track map,
hover readout shows every lap's value and the spread."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF


@pytest.fixture()
def views(qtbot, workspace):
    from iracing_analysis.gui.cursor import CursorController
    from iracing_analysis.gui.readout import HoverReadout
    from iracing_analysis.gui.trackmap_view import TrackMapView
    from iracing_analysis.gui.traces import TraceView

    cursor = CursorController(grid_end=float(workspace.grid[-1]))
    traces = TraceView(workspace, cursor)
    trackmap = TrackMapView(workspace, cursor)
    readout = HoverReadout(workspace, cursor)
    for w in (traces, trackmap, readout):
        qtbot.addWidget(w)
        w.resize(1200, 800)
        w.show()
    return workspace, cursor, traces, trackmap, readout


def test_cursor_synced_across_all_plots_and_map(views) -> None:
    workspace, cursor, traces, trackmap, readout = views
    d = 1234.5
    cursor.set_distance(d)

    # every stacked plot's crosshair at the same distance
    positions = [line.value() for line in traces.cursor_lines]
    assert len(positions) >= 6  # delta + speed + throttle + brake + steering + rpm + gear
    assert all(p == pytest.approx(d) for p in positions)

    # track map marker at the same place on the track
    idx = workspace.index_at(d)
    sx, sy = workspace.track_shape()
    mx, my = trackmap.marker_pos()
    assert mx == pytest.approx(float(sx[idx]))
    assert my == pytest.approx(float(sy[idx]))


def test_cursor_clamped_to_track(views) -> None:
    workspace, cursor, *_ = views
    cursor.set_distance(1e9)
    assert cursor.distance == pytest.approx(float(workspace.grid[-1]))
    cursor.set_distance(-5)
    assert cursor.distance == 0.0


def test_hover_readout_lists_all_laps_and_spread(views) -> None:
    workspace, cursor, traces, trackmap, readout = views
    d = 800.0
    cursor.set_distance(d)
    sample = workspace.cursor_values(d)

    # one row per lap + one spread row
    assert readout.rowCount() == len(workspace.aligned.laps) + 1
    for i in range(len(workspace.aligned.laps)):
        speed_kmh = sample.laps[i]["Speed"] * 3.6
        cell = readout.value_at(i, "Speed")
        assert cell == pytest.approx(speed_kmh, abs=0.05)
    spread_kmh = sample.spread["Speed"] * 3.6
    assert readout.value_at(len(workspace.aligned.laps), "Speed") == pytest.approx(
        spread_kmh, abs=0.05
    )


def test_mouse_move_maps_to_distance(views, qapp) -> None:
    workspace, cursor, traces, trackmap, readout = views
    qapp.processEvents()
    plot = traces.plots[1]  # speed plot
    target = 950.0
    scene_pos = plot.vb.mapViewToScene(QPointF(target, 30.0))
    traces._on_mouse_moved(scene_pos)
    assert cursor.distance == pytest.approx(target, abs=2.0)


def test_main_window_smoke(qtbot, ibt_path, clean_laps, tmp_path) -> None:
    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=ibt_path.parent)
    qtbot.addWidget(win)
    win.show()
    win.refresh_sessions()
    assert win.browser.session_count() == 1

    win.open_laps(clean_laps)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Traces" in tabs and "Track Map" in tabs
    assert win.workspace is not None
    assert len(win.workspace.aligned.laps) == len(clean_laps)
