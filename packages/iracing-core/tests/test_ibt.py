"""Phase 1 gate: .ibt writing + parsing.

The synthetic writer is validated against pyirsdk's own IBT parser (the
oracle), and IbtReader is validated against both.
"""

from __future__ import annotations

import numpy as np


def test_writer_roundtrip_against_pyirsdk_oracle(ibt_path, synthetic_session) -> None:
    """pyirsdk's IBT class must read back exactly what write_ibt wrote."""
    import irsdk

    ibt = irsdk.IBT()
    ibt.open(str(ibt_path))
    try:
        n = ibt._disk_header.session_record_count
        ch = synthetic_session.channels
        assert n == len(ch["SessionTime"])
        assert set(ibt.var_headers_names) == set(ch.keys())

        speed = np.asarray(ibt.get_all("Speed"), dtype=np.float64)
        np.testing.assert_allclose(speed, ch["Speed"].astype(np.float32), rtol=0, atol=1e-6)

        lap = np.asarray(ibt.get_all("Lap"))
        np.testing.assert_array_equal(lap, ch["Lap"])

        # single-sample access
        assert ibt.get(0, "Lap") == int(ch["Lap"][0])
    finally:
        ibt.close()


def test_ibt_reader_channels_match_pyirsdk(ibt_path) -> None:
    import irsdk

    from iracing_core.ibt import IbtReader

    with IbtReader(ibt_path) as reader:
        ibt = irsdk.IBT()
        ibt.open(str(ibt_path))
        try:
            for name in ("Speed", "Lap", "SessionTime", "Brake", "OnPitRoad", "Lat"):
                mine = reader.get_channel(name)
                oracle = np.asarray(ibt.get_all(name))
                np.testing.assert_allclose(
                    mine, oracle, rtol=0, atol=0, err_msg=f"channel {name} mismatch"
                )
        finally:
            ibt.close()


def test_ibt_reader_metadata(ibt_path, synthetic_session) -> None:
    from iracing_core.ibt import IbtReader

    with IbtReader(ibt_path) as reader:
        assert reader.tick_rate == 60
        assert reader.record_count == len(synthetic_session.channels["SessionTime"])
        assert "Speed" in reader.channel_names
        assert reader.has_channel("Speed")
        assert not reader.has_channel("NoSuchChannel")


def test_ibt_reader_session_info_yaml(ibt_path) -> None:
    from iracing_core.ibt import IbtReader

    with IbtReader(ibt_path) as reader:
        info = reader.session_info
        assert info["WeekendInfo"]["TrackDisplayName"] == "Fantasia International"
        assert info["WeekendInfo"]["TrackName"] == "fantasia full"
        drivers = info["DriverInfo"]["Drivers"]
        idx = info["DriverInfo"]["DriverCarIdx"]
        me = next(d for d in drivers if d["CarIdx"] == idx)
        assert me["UserName"] == "Test Driver"
        assert len(info["SplitTimeInfo"]["Sectors"]) == 3


def test_ibt_reader_meta_object(ibt_path) -> None:
    from iracing_core.ibt import IbtReader

    with IbtReader(ibt_path) as reader:
        meta = reader.meta
        assert meta.track_display_name == "Fantasia International"
        assert meta.car_screen_name == "Formula Fable GT3"
        assert meta.driver_name == "Test Driver"
        assert meta.session_type == "Practice"
        assert abs(meta.track_length_m - 3200) < 400  # parsed from '3.xx km'
        assert meta.tick_rate == 60
        assert len(meta.sector_starts_pct) == 3
