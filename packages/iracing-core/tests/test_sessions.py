"""Phase 1 gate: lap splitting and telemetry-folder session browsing."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def laps(ibt_path):
    from iracing_core.ibt import IbtReader
    from iracing_core.sessions import extract_laps

    with IbtReader(ibt_path) as reader:
        return extract_laps(reader)


def test_extract_laps_counts(laps, synthetic_session) -> None:
    # out-lap + 3 flying laps + in-lap
    assert len(laps) == 5
    flying = [lap for lap in laps if lap.is_clean]
    assert len(flying) == 3


def test_lap_times_match_ground_truth(laps, synthetic_session) -> None:
    flying = [lap for lap in laps if lap.is_clean]
    for lap, truth in zip(flying, synthetic_session.lap_times):
        assert lap.lap_time is not None
        # sub-sample S/F interpolation should get us well under one frame
        assert abs(lap.lap_time - truth) < 0.005, (
            f"lap {lap.lap_number}: {lap.lap_time} vs ground truth {truth}"
        )


def test_lap_flags(laps) -> None:
    out, *mid, last = laps
    assert out.is_out_lap and not out.is_clean
    assert last.is_in_lap and not last.is_clean
    for lap in mid:
        assert lap.is_clean and not lap.is_out_lap and not lap.is_in_lap


def test_lap_channel_slices(laps) -> None:
    lap = next(lap for lap in laps if lap.is_clean)
    dist = lap.channel("LapDist")
    speed = lap.channel("Speed")
    assert len(dist) == len(speed) > 1000
    # distance grows monotonically within a lap (modulo sensor noise: none here)
    assert np.all(np.diff(dist) >= 0)
    assert dist[0] < 50.0  # starts at S/F


def test_scan_telemetry_dir_groups_sessions(tmp_path, synthetic_session) -> None:
    from iracing_core.sessions import scan_telemetry_dir
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import build_session, default_track

    # session A: two files from the same iRacing session (same SessionID)
    write_ibt(
        tmp_path / "carA" / "fantasia_a1.ibt",
        channels=synthetic_session.channels,
        session_info=synthetic_session.session_info,
    )
    write_ibt(
        tmp_path / "carA" / "fantasia_a2.ibt",
        channels=synthetic_session.channels,
        session_info=synthetic_session.session_info,
    )
    # session B: different session on another track
    other = build_session(
        track=default_track(),
        n_laps=1,
        seed=7,
        track_display_name="Speedonia Ring",
        track_name="speedonia ring",
        session_id=999,
    )
    write_ibt(
        tmp_path / "carB" / "speedonia_b.ibt",
        channels=other.channels,
        session_info=other.session_info,
    )

    groups = scan_telemetry_dir(tmp_path)
    assert len(groups) == 2
    by_track = {g.meta.track_display_name: g for g in groups}
    assert len(by_track["Fantasia International"].files) == 2
    assert len(by_track["Speedonia Ring"].files) == 1


def test_default_telemetry_dir_shape() -> None:
    from iracing_core.sessions import default_telemetry_dir

    p = default_telemetry_dir()
    assert p.name == "telemetry"
    assert p.parent.name == "iRacing"
