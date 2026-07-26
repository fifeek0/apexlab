"""Phase 4 gate (part 2): corner detection + per-corner time-loss attribution."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def controlled_laps(tmp_path_factory):
    """Two laps with a KNOWN difference: lap 0 = baseline, lap 1 = 4% less
    lateral grip (lower apex speeds everywhere). No random spread."""
    from dataclasses import replace

    from iracing_core import IbtReader, extract_laps
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import DriverParams, build_session, default_track

    base = DriverParams()
    slow = replace(base, a_lat_max=base.a_lat_max * 0.96)
    session = build_session(
        track=default_track(),
        n_laps=2,
        seed=1,
        with_out_in_laps=False,
        per_lap_params=[base, slow],
        param_spread=0.0,
    )
    path = tmp_path_factory.mktemp("controlled") / "controlled.ibt"
    write_ibt(path, channels=session.channels, session_info=session.session_info)
    with IbtReader(path) as reader:
        laps = [lap for lap in extract_laps(reader) if lap.is_complete]
    return session, laps


@pytest.fixture(scope="module")
def controlled_workspace(controlled_laps):
    from iracing_analysis.analysis.workspace import Workspace

    _, laps = controlled_laps
    return Workspace(laps, reference_index=0)


def test_corner_detection_matches_track_layout(controlled_laps, controlled_workspace) -> None:
    from iracing_analysis.analysis.corners import detect_corners

    session, _ = controlled_laps
    corners = detect_corners(controlled_workspace.reference)
    assert len(corners) == session.track.n_corners

    detected = np.array([c.apex_m for c in corners])
    truth = np.array(sorted(c.apex_s for c in session.track.corners))
    # each detected apex within 30 m of a ground-truth arc midpoint
    for apex in detected:
        assert np.min(np.abs(truth - apex)) < 30.0, f"apex at {apex:.0f} m unexpected"


def test_corner_windows_are_disjoint_and_ordered(controlled_workspace) -> None:
    from iracing_analysis.analysis.corners import detect_corners

    corners = detect_corners(controlled_workspace.reference)
    for a, b in zip(corners, corners[1:]):
        assert a.start_m < a.apex_m < a.end_m
        assert a.end_m <= b.start_m


def test_segment_deltas_sum_to_total(controlled_workspace) -> None:
    """Acceptance: per-corner (+ straight) deltas sum to the total delta."""
    from iracing_analysis.analysis.corners import detect_corners, segment_deltas

    ws = controlled_workspace
    corners = detect_corners(ws.reference)
    segments = segment_deltas(ws, lap_index=1, corners=corners)

    total = float(ws.deltas[1][-1] - ws.deltas[1][0])
    assert sum(seg.delta for seg in segments) == pytest.approx(total, abs=1e-6)

    # the segments tile the whole lap without gaps
    assert segments[0].start_m == 0.0
    for a, b in zip(segments, segments[1:]):
        assert a.end_m == pytest.approx(b.start_m)
    assert segments[-1].end_m == pytest.approx(float(ws.grid[-1]))


def test_attribution_flags_lower_apex_speed(controlled_workspace) -> None:
    """Lap 1 has 4% less grip: the top time-loss corners must be attributed
    to lower minimum/apex speed."""
    from iracing_analysis.analysis.corners import analyze_corners

    comparisons = analyze_corners(controlled_workspace, lap_index=1)
    assert comparisons, "no corners analysed"

    # overall the lap is slower, and corner deltas dominate the loss
    lost = [c for c in comparisons if c.delta_total > 0.005]
    assert len(lost) >= 6  # grip affects every real corner

    ranked = sorted(comparisons, key=lambda c: c.delta_total, reverse=True)
    for comp in ranked[:5]:
        assert comp.metrics.min_speed < comp.ref_metrics.min_speed
        assert any("apex speed" in reason for reason in comp.reasons), comp.reasons


def test_attribution_flags_exit_loss_for_weak_engine(tmp_path) -> None:
    """Lap 1 with 12% weaker engine: losses concentrate on corner exits."""
    from dataclasses import replace

    from iracing_core import IbtReader, extract_laps
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import DriverParams, build_session, default_track

    from iracing_analysis.analysis.corners import analyze_corners
    from iracing_analysis.analysis.workspace import Workspace

    base = DriverParams()
    weak = replace(base, a_engine=base.a_engine * 0.88)
    session = build_session(
        track=default_track(),
        n_laps=2,
        seed=2,
        with_out_in_laps=False,
        per_lap_params=[base, weak],
        param_spread=0.0,
    )
    path = tmp_path / "weak_engine.ibt"
    write_ibt(path, channels=session.channels, session_info=session.session_info)
    with IbtReader(path) as reader:
        laps = [lap for lap in extract_laps(reader) if lap.is_complete]
    ws = Workspace(laps, reference_index=0)

    comparisons = analyze_corners(ws, lap_index=1)

    # acceleration losses live on exits + the straights they feed (delta_carry),
    # not in the braking phase
    entry_loss = sum(c.delta_entry for c in comparisons)
    exit_and_carry = sum(c.delta_exit + c.delta_carry for c in comparisons)
    assert exit_and_carry > 2.0 * max(entry_loss, 0.0), (entry_loss, exit_and_carry)

    # and the 'why' points at throttle/exit for the worst corners
    from iracing_analysis.analysis.corners import rank_by_time_lost

    top = rank_by_time_lost(comparisons)[:5]
    throttle_flagged = [
        c
        for c in top
        if any("full throttle" in r or "following straight" in r for r in c.reasons)
    ]
    assert len(throttle_flagged) >= 3, [c.reasons for c in top]


def test_corner_ranking_report(controlled_workspace) -> None:
    from iracing_analysis.analysis.corners import analyze_corners, rank_by_time_lost

    comparisons = analyze_corners(controlled_workspace, lap_index=1)
    ranked = rank_by_time_lost(comparisons)
    losses = [c.time_lost for c in ranked]
    assert losses == sorted(losses, reverse=True)
    assert all(c.reasons for c in ranked)
