"""LiveComparison: real-time delta vs a reference lap from the library."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ref_and_stream(tmp_path_factory):
    """Reference lap + a live stream of a slower lap (4% less grip)."""
    from dataclasses import replace

    from iracing_core import IbtReader, extract_laps
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import DriverParams, build_session, default_track

    base = DriverParams()
    slow = replace(base, a_lat_max=base.a_lat_max * 0.96)
    session = build_session(
        track=default_track(), n_laps=2, seed=21, with_out_in_laps=False,
        per_lap_params=[base, slow], param_spread=0.0,
    )
    path = tmp_path_factory.mktemp("livecmp") / "s.ibt"
    write_ibt(path, channels=session.channels, session_info=session.session_info)
    with IbtReader(path) as reader:
        laps = [lap for lap in extract_laps(reader) if lap.is_complete]
    reference, slow_lap = laps[0], laps[1]

    ch = session.channels
    lap_arr = ch["Lap"]
    idx = np.flatnonzero(lap_arr == slow_lap.lap_number)
    names = list(ch)
    stream = [{n: ch[n][i] for n in names} for i in idx]
    return reference, slow_lap, stream


def test_live_delta_tracks_reference(ref_and_stream) -> None:
    from iracing_core.live_compare import LiveComparison

    reference, slow_lap, stream = ref_and_stream
    cmp = LiveComparison(reference)

    states = [cmp.feed(sample) for sample in stream]
    states = [s for s in states if s is not None]
    assert len(states) > 1000

    # delta starts near zero and grows to ~the lap-time difference
    expected = slow_lap.lap_time - reference.lap_time
    assert abs(states[10].time_delta) < 0.15
    assert states[-1].time_delta == pytest.approx(expected, abs=0.10)
    # delta never jumps wildly between consecutive samples
    deltas = np.array([s.time_delta for s in states])
    assert np.abs(np.diff(deltas)).max() < 0.25


def test_live_state_fields(ref_and_stream) -> None:
    from iracing_core.live_compare import LiveComparison

    reference, _, stream = ref_and_stream
    cmp = LiveComparison(reference)
    state = None
    for sample in stream[: len(stream) // 3]:
        state = cmp.feed(sample) or state

    assert state is not None
    assert 0 <= state.lap_dist_m <= cmp.track_length_m
    assert state.ref_gear in range(1, 8)
    assert -80.0 < state.speed_delta_kmh < 80.0
    assert 0.0 <= state.ref_throttle <= 1.0
    assert 0.0 <= state.ref_brake <= 1.0


def test_braking_zones_from_reference(ref_and_stream) -> None:
    from iracing_core.live_compare import LiveComparison

    reference, _, stream = ref_and_stream
    cmp = LiveComparison(reference)
    # synthetic track has 10 braking corners
    assert 8 <= len(cmp.braking_zones_m) <= 12
    assert all(0 <= z < cmp.track_length_m for z in cmp.braking_zones_m)

    # distance to the next zone decreases while approaching it
    dists = []
    for sample in stream[:400]:
        state = cmp.feed(sample)
        if state is not None:
            dists.append(state.next_braking_m)
    drops = np.diff([d for d in dists if d is not None])
    assert (drops <= 1e-6).mean() > 0.9  # decreasing except at zone hand-off


def test_reference_preview_window(ref_and_stream) -> None:
    """The overlay needs the reference's upcoming inputs (Bloops-style
    preview): ask for a window around the current position."""
    from iracing_core.live_compare import LiveComparison

    reference, _, stream = ref_and_stream
    cmp = LiveComparison(reference)
    for sample in stream[:600]:
        cmp.feed(sample)

    window = cmp.reference_window(behind_m=250.0, ahead_m=150.0)
    assert window is not None
    x, throttle, brake = window.offsets_m, window.throttle, window.brake
    assert x[0] >= -250.0 and x[-1] <= 150.0
    assert len(x) == len(throttle) == len(brake) > 100
    assert 0.0 <= throttle.min() and throttle.max() <= 1.0
