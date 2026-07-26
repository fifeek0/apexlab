"""Workspace: the UI-agnostic analysis state shared by all views."""

from __future__ import annotations

import numpy as np
import pytest


def test_workspace_aligns_laps(workspace, clean_laps) -> None:
    assert len(workspace.aligned.laps) == len(clean_laps)
    assert workspace.grid[0] == 0.0
    assert workspace.reference is workspace.aligned.laps[0]


def test_workspace_deltas_match_core(workspace) -> None:
    from iracing_core import delta_time

    deltas = workspace.deltas
    assert np.allclose(deltas[0], 0.0)
    for i, al in enumerate(workspace.aligned.laps):
        np.testing.assert_allclose(deltas[i], delta_time(al, workspace.reference))


def test_workspace_reference_switch(workspace) -> None:
    workspace.set_reference(1)
    assert workspace.reference is workspace.aligned.laps[1]
    assert np.allclose(workspace.deltas[1], 0.0)
    assert not np.allclose(workspace.deltas[0], 0.0)


def test_cursor_values(workspace) -> None:
    d = float(workspace.grid[len(workspace.grid) // 3])
    sample = workspace.cursor_values(d)
    assert sample.distance == pytest.approx(d)
    idx = sample.index
    for i, al in enumerate(workspace.aligned.laps):
        assert sample.laps[i]["Speed"] == pytest.approx(float(al.channels["Speed"][idx]))
        assert sample.laps[i]["Delta"] == pytest.approx(float(workspace.deltas[i][idx]))
    # spread across laps
    speeds = [row["Speed"] for row in sample.laps]
    assert sample.spread["Speed"] == pytest.approx(max(speeds) - min(speeds))


def test_track_shape(workspace, synthetic_session) -> None:
    x, y = workspace.track_shape()
    assert len(x) == len(workspace.grid)
    # compare with the generator's ground-truth geometry (both mean-centred;
    # the projections differ only by the reference origin)
    gx, gy = synthetic_session.track.xy(workspace.grid)
    np.testing.assert_allclose(x - x.mean(), gx - gx.mean(), atol=3.0)
    np.testing.assert_allclose(y - y.mean(), gy - gy.mean(), atol=3.0)
