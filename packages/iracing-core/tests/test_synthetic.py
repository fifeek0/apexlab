"""Sanity checks for the synthetic track / telemetry generator.

Every later phase's tests rest on this generator, so its physics must be
credible: closed track, plausible speeds, consistent ground-truth lap data.
"""

from __future__ import annotations

import numpy as np


def test_default_track_is_closed() -> None:
    from iracing_core.testing.synthetic import default_track

    track = default_track()
    assert track.length > 2000.0  # a real circuit, not a kart track
    x0, y0 = track.xy(0.0)
    x1, y1 = track.xy(track.length)
    assert abs(x1 - x0) < 1.0 and abs(y1 - y0) < 1.0  # geometric closure


def test_default_track_has_known_corners() -> None:
    from iracing_core.testing.synthetic import default_track

    track = default_track()
    assert track.n_corners >= 8
    # every corner must force a lift: v_limit below straight-line speeds
    for corner in track.corners:
        assert corner.radius <= 80.0
        assert 0.0 < corner.apex_s < track.length


def test_session_channels_are_consistent(synthetic_session) -> None:
    ch = synthetic_session.channels
    n = len(ch["SessionTime"])
    for name, arr in ch.items():
        assert len(arr) == n, f"channel {name} length mismatch"

    # 60 Hz sampling
    dt = np.diff(ch["SessionTime"])
    assert np.allclose(dt, 1.0 / 60.0, atol=1e-9)

    speed = ch["Speed"]
    assert speed.min() >= 0.0
    assert 40.0 < speed.max() < 120.0  # m/s: fast but physical

    # LapDistPct within [0, 1); Lap increments by 0 or 1
    assert ch["LapDistPct"].min() >= 0.0 and ch["LapDistPct"].max() < 1.0
    lap_steps = np.diff(ch["Lap"])
    assert set(np.unique(lap_steps)).issubset({0, 1})

    # throttle/brake in [0, 1] and never hard-overlapping
    assert ch["Throttle"].min() >= 0.0 and ch["Throttle"].max() <= 1.0
    assert ch["Brake"].min() >= 0.0 and ch["Brake"].max() <= 1.0


def test_session_ground_truth_lap_times(synthetic_session) -> None:
    sess = synthetic_session
    assert len(sess.lap_times) == 3  # flying laps only
    for lt in sess.lap_times:
        assert 60.0 < lt < 180.0
    # laps differ (different driver params per lap) but are in the same class
    assert max(sess.lap_times) - min(sess.lap_times) > 0.05
    assert max(sess.lap_times) - min(sess.lap_times) < 10.0


def test_lat_accel_matches_curvature(synthetic_session) -> None:
    ch = synthetic_session.channels
    v = ch["Speed"]
    lat = ch["LatAccel"]
    # somewhere mid-corner lateral acceleration must be significant
    assert np.abs(lat).max() > 10.0
    # and never beyond the generator's grip model (~plus margin)
    assert np.abs(lat).max() < 35.0
    # straights: near-zero lateral accel at top speed
    top = v > np.percentile(v, 99)
    assert np.abs(lat[top]).mean() < 2.0
