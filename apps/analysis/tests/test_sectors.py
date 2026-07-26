"""Phase 4 gate (part 1): official sectors and configurable mini-sectors."""

from __future__ import annotations

import numpy as np
import pytest


def test_official_sector_boundaries_from_yaml(workspace) -> None:
    from iracing_analysis.analysis.sectors import official_sector_boundaries

    bounds = official_sector_boundaries(workspace)
    # synthetic session ships 3 official sectors at 0 / 0.31 / 0.66
    track_len = workspace.laps[0].meta.track_length_m
    assert len(bounds) == 3
    assert bounds[0] == 0.0
    assert bounds[1] == pytest.approx(0.31 * track_len, rel=0.05)


def test_sector_times_sum_to_lap_time(workspace) -> None:
    from iracing_analysis.analysis.sectors import (
        official_sector_boundaries,
        sector_times,
    )

    bounds = official_sector_boundaries(workspace)
    table = sector_times(workspace, bounds)
    assert table.times.shape == (len(workspace.laps), 3)
    for i, al in enumerate(workspace.aligned.laps):
        assert table.times[i].sum() == pytest.approx(al.lap.lap_time, abs=0.002)


def test_minisector_times_and_best(workspace) -> None:
    from iracing_analysis.analysis.sectors import minisector_boundaries, sector_times

    bounds = minisector_boundaries(workspace, count=20)
    assert len(bounds) == 20
    table = sector_times(workspace, bounds)
    assert table.times.shape == (len(workspace.laps), 20)

    # each lap's minisectors sum to its lap time
    for i, al in enumerate(workspace.aligned.laps):
        assert table.times[i].sum() == pytest.approx(al.lap.lap_time, abs=0.002)

    # best-lap-per-sector really is the minimum
    for k in range(20):
        best = table.best_lap[k]
        assert table.times[best, k] == pytest.approx(table.times[:, k].min())
