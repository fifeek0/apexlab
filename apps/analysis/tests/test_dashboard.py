"""Single-page pit-wall dashboard: map + traces + corners + auto coach."""

from __future__ import annotations

import pytest

from iracing_analysis.insights.base import InsightProvider, InsightResult


class FakeCoach(InsightProvider):
    name = "fake-coach"

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, summary: dict) -> InsightResult:
        self.calls.append(summary)
        return InsightResult(
            ok=True,
            text="## Werdykt\nUżywaj trail brakingu w T3.",
            provider=self.name,
            model="fake",
        )


@pytest.fixture()
def dashboard(qtbot, workspace):
    from iracing_analysis.gui.dashboard import AnalysisDashboard

    dash = AnalysisDashboard(workspace, provider=FakeCoach(), autostart_coach=False)
    qtbot.addWidget(dash)
    dash.resize(1600, 1000)
    dash.show()
    return dash


def test_dashboard_is_single_page(dashboard, workspace) -> None:
    # all the instruments live on one page
    assert dashboard.trackmap is not None
    assert dashboard.traces is not None
    assert dashboard.readout is not None
    assert dashboard.corners is not None
    assert dashboard.coach is not None
    # header carries the essential comparison
    text = dashboard.header_text()
    assert "Fantasia International" in text


def test_dashboard_cursor_links_map_and_traces(dashboard, workspace) -> None:
    d = 1500.0
    dashboard.cursor.set_distance(d)
    idx = workspace.index_at(d)
    x, y = workspace.track_shape()
    mx, my = dashboard.trackmap.marker_pos()
    assert mx == pytest.approx(float(x[idx]))
    assert my == pytest.approx(float(y[idx]))
    assert all(line.value() == pytest.approx(d) for line in dashboard.traces.cursor_lines)


def test_dashboard_map_has_corner_labels_and_delta_coloring(dashboard) -> None:
    labels = dashboard.trackmap.corner_labels()
    assert len(labels) >= 8  # one per detected corner
    assert labels[0].startswith("T")
    assert dashboard.trackmap.color_mode == "delta"


def test_workspace_lap_lines_share_one_frame(workspace) -> None:
    """Per-lap GPS driving lines projected into a common local frame — the
    reference line must coincide with the track shape used by the map."""
    import numpy as np

    lines = workspace.lap_lines()
    assert len(lines) == len(workspace.laps)
    sx, sy = workspace.track_shape()
    ref_line = lines[workspace.reference_index]
    assert ref_line is not None
    np.testing.assert_allclose(ref_line[0], sx, atol=1e-6)
    np.testing.assert_allclose(ref_line[1], sy, atol=1e-6)


def test_map_draws_a_driving_line_per_lap(dashboard, workspace) -> None:
    assert len(dashboard.trackmap.lap_line_items) == len(workspace.laps)


def test_coach_panel_generates_detailed_report(dashboard) -> None:
    provider = dashboard.coach.provider
    result = dashboard.coach.generate_sync()
    assert result.ok
    assert "trail brakingu" in dashboard.coach.report_text()
    # the coach receives the full detailed summary, not a stub
    summary = provider.calls[0]
    assert "corners" in summary and "theoretical_best" in summary


def test_coach_panel_disabled_message(qtbot, workspace) -> None:
    from iracing_analysis.gui.dashboard import CoachPanel

    panel = CoachPanel(workspace, provider=None, autostart=False)
    qtbot.addWidget(panel)
    assert "AI" in panel.status_text()  # explains how to enable


def test_main_window_opens_pit_wall_first(qtbot, clean_laps, tmp_path) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=str(tmp_path), library=LapLibrary(tmp_path / "lib"))
    qtbot.addWidget(win)
    win.open_laps(clean_laps)
    assert win.tabs.tabText(0) == "Pit Wall"
    assert win.tabs.currentIndex() == 0
