"""Phase 6 gate (analysis part): theoretical best lap."""

from __future__ import annotations

import pytest


def test_theoretical_best_not_slower_than_actual(workspace) -> None:
    from iracing_analysis.analysis.theoretical import theoretical_best

    tb = theoretical_best(workspace, minisectors=20)
    actual_best = min(lap.lap_time for lap in workspace.laps)
    assert tb.total <= actual_best + 1e-9
    assert tb.actual_best == pytest.approx(actual_best)
    assert tb.gap == pytest.approx(actual_best - tb.total)
    assert tb.gap >= 0.0
    assert len(tb.best_times) == 20
    assert tb.total == pytest.approx(float(tb.best_times.sum()))


def test_theoretical_best_improves_with_refinement(workspace) -> None:
    """A 2x-refined partition can only mix laps more freely."""
    from iracing_analysis.analysis.theoretical import theoretical_best

    coarse = theoretical_best(workspace, minisectors=10)
    fine = theoretical_best(workspace, minisectors=20)
    assert fine.total <= coarse.total + 1e-9


def test_theoretical_best_contributors_valid(workspace) -> None:
    from iracing_analysis.analysis.theoretical import theoretical_best

    tb = theoretical_best(workspace, minisectors=15)
    assert set(tb.contributors.tolist()).issubset(set(range(len(workspace.laps))))


def test_sectors_view_shows_theoretical_best(qtbot, workspace) -> None:
    from iracing_analysis.gui.sectors_view import SectorsView

    view = SectorsView(workspace, minisector_count=12)
    qtbot.addWidget(view)
    text = view.theoretical_label.text()
    assert "Theoretical best" in text
    assert "optimistic" in text.lower()
