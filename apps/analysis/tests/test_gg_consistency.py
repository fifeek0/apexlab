"""Phase 5 gate: GG diagram within physical range + consistency metrics."""

from __future__ import annotations

import numpy as np
import pytest


def test_gg_points_in_physical_range(workspace) -> None:
    from iracing_analysis.analysis.gg import gg_points

    for al in workspace.aligned.laps:
        gg = gg_points(al)
        assert gg.lat_g.shape == gg.long_g.shape == workspace.grid.shape
        combined = np.hypot(gg.lat_g, gg.long_g)
        # a GT3-class car: real grip but nothing unphysical
        assert 1.5 < np.abs(gg.lat_g).max() < 3.5
        assert 0.5 < np.abs(gg.long_g).max() < 3.5
        assert combined.max() < 3.6


def test_gg_stats(workspace) -> None:
    from iracing_analysis.analysis.gg import gg_points, gg_stats

    stats = gg_stats(gg_points(workspace.reference))
    assert stats["max_lat_g"] > 1.5
    assert stats["max_brake_g"] > 1.0
    assert 0.0 < stats["combined_p95_g"] <= stats["max_combined_g"] < 3.6


def test_lap_time_stats(workspace) -> None:
    from iracing_analysis.analysis.consistency import lap_time_stats

    stats = lap_time_stats(workspace.laps)
    times = [lap.lap_time for lap in workspace.laps]
    assert stats["best"] == pytest.approx(min(times))
    assert stats["mean"] == pytest.approx(float(np.mean(times)))
    assert stats["std"] > 0.0
    assert stats["count"] == len(times)


def test_channel_histograms(workspace) -> None:
    from iracing_analysis.analysis.consistency import channel_histograms

    hist = channel_histograms(workspace, "Throttle", bins=10)
    assert hist.counts.shape == (len(workspace.laps), 10)
    assert len(hist.edges) == 11
    # every grid point lands in exactly one bin
    assert np.all(hist.counts.sum(axis=1) == len(workspace.grid))


def test_cross_lap_variability(workspace) -> None:
    from iracing_analysis.analysis.consistency import cross_lap_variability

    var = cross_lap_variability(workspace)
    assert var.speed_std.shape == workspace.grid.shape
    assert np.all(var.speed_std >= 0.0)
    # synthetic laps are similar: repeatable driving, small dispersion
    assert 0.0 < var.speed_std.mean() < 2.0
    assert 0.0 <= var.throttle_std.mean() < 0.2


def test_gui_phase5_views(qtbot, workspace, clean_laps) -> None:
    from iracing_analysis.gui.consistency_view import ConsistencyView
    from iracing_analysis.gui.cursor import CursorController
    from iracing_analysis.gui.gg_view import GGView
    from iracing_analysis.gui.main_window import MainWindow
    from iracing_analysis.gui.trackmap_view import TrackMapView

    cursor = CursorController(grid_end=float(workspace.grid[-1]))
    gg = GGView(workspace)
    qtbot.addWidget(gg)
    assert gg.point_items, "GG scatter must contain one item per lap"

    cons = ConsistencyView(workspace)
    qtbot.addWidget(cons)

    tm = TrackMapView(workspace, cursor)
    qtbot.addWidget(tm)
    tm.set_color_mode("speed")
    assert tm.color_mode == "speed"
    tm.set_color_mode("delta")
    assert tm.color_mode == "delta"

    win = MainWindow(telemetry_dir=".")
    qtbot.addWidget(win)
    win.open_laps(clean_laps)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "GG Diagram" in tabs and "Consistency" in tabs
