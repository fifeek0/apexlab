"""Phase 2 gate: distance-based alignment + delta-time engine."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def clean_laps(ibt_path):
    from iracing_core.ibt import IbtReader
    from iracing_core.sessions import extract_laps

    with IbtReader(ibt_path) as reader:
        return [lap for lap in extract_laps(reader) if lap.is_clean]


@pytest.fixture(scope="module")
def aligned(clean_laps):
    from iracing_core.alignment import align_laps

    return align_laps(clean_laps, spacing=1.0)


def test_common_grid(aligned, clean_laps) -> None:
    grid = aligned.grid
    assert grid[0] == 0.0
    assert np.allclose(np.diff(grid), 1.0)
    track_len = clean_laps[0].meta.track_length_m
    assert track_len * 0.97 <= grid[-1] <= track_len

    for al in aligned.laps:
        assert al.channels["Speed"].shape == grid.shape
        assert not np.isnan(al.channels["Speed"]).any()
        assert al.time_at.shape == grid.shape


def test_time_at_anchored_to_crossings(aligned) -> None:
    for al in aligned.laps:
        assert al.time_at[0] == pytest.approx(0.0, abs=1e-9)
        # elapsed time is strictly increasing along the lap
        assert np.all(np.diff(al.time_at) > 0)
        # total elapsed ~ lap time (grid stops just short of full length)
        assert al.time_at[-1] == pytest.approx(al.lap.lap_time, abs=0.1)


def test_delta_to_self_is_zero(aligned) -> None:
    from iracing_core.alignment import delta_time

    ref = aligned.laps[0]
    d = delta_time(ref, ref)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_delta_at_finish_matches_lap_time_difference(aligned) -> None:
    """The Phase 2 acceptance test: cumulative delta at S/F equals the
    lap-time difference to within a few milliseconds."""
    from iracing_core.alignment import delta_time

    ref = aligned.laps[0]
    for al in aligned.laps[1:]:
        d = delta_time(al, ref)
        assert d[0] == pytest.approx(0.0, abs=1e-9)
        expected = al.lap.lap_time - ref.lap.lap_time
        # extrapolate the last grid point to the full lap: time anchors make
        # the value at the final grid point already agree to sub-ms
        assert d[-1] == pytest.approx(expected, abs=0.003)


def test_integer_channels_stay_integral(aligned) -> None:
    for al in aligned.laps:
        gear = al.channels["Gear"]
        assert np.array_equal(gear, np.round(gear))
        assert gear.min() >= 1 and gear.max() <= 7


def test_speed_resampling_is_faithful(aligned, clean_laps) -> None:
    """Resampled speed at the original sample distances matches the raw data."""
    al = aligned.laps[0]
    lap = clean_laps[0]
    dist = lap.channel("LapDist").astype(float)
    speed = lap.channel("Speed").astype(float)
    inside = (dist > 5.0) & (dist < aligned.grid[-1] - 5.0)
    resampled_back = np.interp(dist[inside], aligned.grid, al.channels["Speed"])
    # 1 m grid vs ~1.1 m sample spacing: interpolation error must be tiny
    assert np.abs(resampled_back - speed[inside]).max() < 0.15
