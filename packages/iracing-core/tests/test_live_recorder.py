"""LiveLapRecorder: builds complete LapData from a live sample stream."""

from __future__ import annotations

import numpy as np
import pytest


def _stream(channels: dict, indices) -> list[dict]:
    names = list(channels)
    return [{name: channels[name][i] for name in names} for i in indices]


def test_recorder_reproduces_extracted_laps(synthetic_session, ibt_path) -> None:
    from iracing_core import IbtReader, extract_laps
    from iracing_core.live import LiveLapRecorder

    with IbtReader(ibt_path) as reader:
        expected = extract_laps(reader)
        meta = reader.meta

    ch = synthetic_session.channels
    recorder = LiveLapRecorder(meta=meta)
    closed = []
    for sample in _stream(ch, range(len(ch["SessionTime"]))):
        lap = recorder.feed(sample)
        if lap is not None:
            closed.append(lap)
    tail = recorder.finish()
    if tail is not None:
        closed.append(tail)

    assert len(closed) == len(expected)
    for got, want in zip(closed, expected):
        assert got.lap_number == want.lap_number
        if want.lap_time is None:
            assert got.lap_time is None
        else:
            assert got.lap_time == pytest.approx(want.lap_time, abs=0.005)
        assert got.is_clean == want.is_clean
        np.testing.assert_allclose(got.channel("Speed"), want.channel("Speed"))
    # metadata travels with each live lap
    assert closed[0].meta is meta


def test_recorder_lower_sample_rate_still_times_laps(synthetic_session, ibt_path) -> None:
    """Polling the sim at ~20 Hz (every 3rd sample) must still produce
    accurate lap times thanks to crossing interpolation."""
    from iracing_core import IbtReader, extract_laps
    from iracing_core.live import LiveLapRecorder

    with IbtReader(ibt_path) as reader:
        expected = [lap for lap in extract_laps(reader) if lap.is_clean]

    ch = synthetic_session.channels
    recorder = LiveLapRecorder()
    closed = []
    for sample in _stream(ch, range(0, len(ch["SessionTime"]), 3)):
        lap = recorder.feed(sample)
        if lap is not None:
            closed.append(lap)
    clean = [lap for lap in closed if lap.is_clean]
    assert len(clean) == len(expected)
    for got, want in zip(clean, expected):
        assert got.lap_time == pytest.approx(want.lap_time, abs=0.05)
