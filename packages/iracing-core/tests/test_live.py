"""Live-ingest wrapper tests (via pyirsdk's test_file replay path)."""

from __future__ import annotations


def test_live_telemetry_snapshot_from_test_file(ibt_path, synthetic_session) -> None:
    from iracing_core.live import LiveTelemetry

    live = LiveTelemetry()
    assert live.connect(test_file=str(ibt_path))
    try:
        snap = live.snapshot(["Speed", "Lap", "LapDist"])
        assert set(snap) == {"Speed", "Lap", "LapDist"}
        # test_file mode exposes the first record of the data block
        assert abs(snap["Speed"] - float(synthetic_session.channels["Speed"][0])) < 1e-4
        assert snap["Lap"] == int(synthetic_session.channels["Lap"][0])
        assert live.is_connected
    finally:
        live.disconnect()
    assert not live.is_connected


def test_live_telemetry_graceful_when_sim_absent() -> None:
    from iracing_core.live import LiveTelemetry

    live = LiveTelemetry()
    # no sim running on this machine (and no Windows shared memory at all)
    assert live.connect() is False
    assert not live.is_connected
    live.disconnect()  # must not raise
