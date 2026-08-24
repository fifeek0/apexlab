"""Garage 61 CSV lap import: single-lap telemetry without SessionTime/LapDist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

G61_COLUMNS = [
    "Speed", "LapDistPct", "Lat", "Lon", "Brake", "Throttle", "RPM",
    "SteeringWheelAngle", "Gear", "Clutch", "ABSActive", "DRSActive",
    "LatAccel", "LongAccel", "VertAccel", "Yaw", "YawRate", "PositionType",
]


def _write_g61_csv(path: Path, lap, lap_time: float) -> Path:
    """Render one of our synthetic laps in Garage 61's export format."""
    import pandas as pd

    n = lap.n_samples
    frame = pd.DataFrame(
        {
            "Speed": lap.channel("Speed"),
            "LapDistPct": lap.channel("LapDistPct"),
            "Lat": lap.channel("Lat"),
            "Lon": lap.channel("Lon"),
            "Brake": lap.channel("Brake"),
            "Throttle": lap.channel("Throttle"),
            "RPM": lap.channel("RPM"),
            "SteeringWheelAngle": lap.channel("SteeringWheelAngle"),
            "Gear": lap.channel("Gear"),
            "Clutch": np.ones(n, dtype=np.float32),
            "ABSActive": ["false"] * n,
            "DRSActive": ["false"] * n,
            "LatAccel": lap.channel("LatAccel"),
            "LongAccel": lap.channel("LongAccel"),
            "VertAccel": np.full(n, 9.81, dtype=np.float32),
            "Yaw": np.zeros(n, dtype=np.float32),
            "YawRate": np.zeros(n, dtype=np.float32),
            "PositionType": np.full(n, 3, dtype=np.int32),
        }
    )[G61_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def g61_csv(tmp_path_factory, ibt_path, synthetic_session):
    from iracing_core import IbtReader, extract_laps

    with IbtReader(ibt_path) as reader:
        clean = [lap for lap in extract_laps(reader) if lap.is_clean]
    lap = clean[0]
    mins, secs = divmod(lap.lap_time, 60.0)
    name = (
        f"Garage 61 - Test Driver - Formula Fable GT3 - Fantasia International "
        f"(Grand Prix) - {int(mins):02d}.{secs:06.3f} - 01FAKEULID0000000000000000.csv"
    )
    path = tmp_path_factory.mktemp("g61") / name
    return _write_g61_csv(path, lap, lap.lap_time), lap


def test_filename_metadata_parse() -> None:
    from iracing_core.garage61 import parse_garage61_filename

    info = parse_garage61_filename(
        "Garage 61 - Mitchell McLeod - Toyota GR86 - Circuit de Spa-Francorchamps "
        "(Grand Prix Pits) - 02.42.795 - 01KXG5HFAB6EPNJ7BMMFRV13JB.csv"
    )
    assert info["driver"] == "Mitchell McLeod"
    assert info["car"] == "Toyota GR86"
    assert info["track"] == "Circuit de Spa-Francorchamps (Grand Prix Pits)"
    assert info["lap_time"] == pytest.approx(162.795)
    assert info["lap_id"] == "01KXG5HFAB6EPNJ7BMMFRV13JB"

    # arbitrary filenames must not crash the reader
    from iracing_core.garage61 import parse_garage61_filename as parse

    assert parse("something_else.csv") == {}


def test_read_garage61_csv_reconstructs_lap(g61_csv, synthetic_session) -> None:
    from iracing_core.garage61 import read_garage61_csv

    path, original = g61_csv
    lap = read_garage61_csv(path)

    assert lap.is_complete
    assert lap.lap_time == pytest.approx(original.lap_time, abs=0.02)
    assert lap.meta.driver_name == "Test Driver"
    assert lap.meta.car_screen_name == "Formula Fable GT3"
    assert "Fantasia International" in lap.meta.track_display_name

    # reconstructed axes
    time = lap.channel("SessionTime")
    assert np.allclose(np.diff(time), 1.0 / 60.0)
    dist = lap.channel("LapDist")
    assert np.all(np.diff(dist) >= 0)
    # track length estimated from speed integration within 1%
    assert lap.meta.track_length_m == pytest.approx(synthetic_session.track.length, rel=0.01)

    np.testing.assert_allclose(lap.channel("Speed"), original.channel("Speed"), atol=1e-4)


def test_csv_lap_aligns_against_ibt_lap(g61_csv, ibt_path) -> None:
    """The money test: a Garage 61 reference lap must work in the delta engine
    against a lap parsed from .ibt."""
    from iracing_core import IbtReader, align_laps, delta_time, extract_laps
    from iracing_core.garage61 import read_garage61_csv

    path, original = g61_csv
    csv_lap = read_garage61_csv(path)

    with IbtReader(ibt_path) as reader:
        clean = [lap for lap in extract_laps(reader) if lap.is_clean]
    other = clean[1]  # a different lap than the one exported to CSV

    aligned = align_laps([csv_lap, other])
    d = delta_time(aligned.laps[1], aligned.laps[0])
    expected = other.lap_time - csv_lap.lap_time
    assert d[-1] == pytest.approx(expected, abs=0.02)


def test_truncated_export_uses_official_time(g61_csv, tmp_path, ibt_path) -> None:
    """Some Garage 61 exports miss the last few % of telemetry; the filename
    lap time (official timing) must win and full-lap delta must still work."""
    import pandas as pd

    from iracing_core import IbtReader, align_laps, delta_time, extract_laps
    from iracing_core.garage61 import read_garage61_csv

    path, original = g61_csv
    frame = pd.read_csv(path)
    cut = int(len(frame) * 0.96)  # drop the final ~4% of samples
    truncated = tmp_path / path.name
    frame.iloc[:cut].to_csv(truncated, index=False)

    lap = read_garage61_csv(truncated)
    assert lap.lap_time == pytest.approx(original.lap_time, abs=0.002)  # filename time
    assert lap.is_complete  # officially timed lap, telemetry merely truncated

    with IbtReader(ibt_path) as reader:
        other = [x for x in extract_laps(reader) if x.is_clean][1]
    aligned = align_laps([lap, other])
    d = delta_time(aligned.laps[1], aligned.laps[0])
    assert d[-1] == pytest.approx(other.lap_time - lap.lap_time, abs=0.15)


def test_mismatched_telemetry_is_rejected(g61_csv, tmp_path) -> None:
    """A file whose data duration equals the official lap time while the
    distance coverage falls short is physically impossible (the telemetry
    stream does not belong to the labelled lap) — the importer must refuse
    it instead of producing corrupted references."""
    import pandas as pd

    from iracing_core.garage61 import read_garage61_csv

    path, original = g61_csv
    frame = pd.read_csv(path)
    cut = int(len(frame) * 0.96)
    truncated = frame.iloc[:cut]

    # filename claims a lap time equal to the truncated data duration
    fake_time = cut / 60.0
    mins, secs = divmod(fake_time, 60.0)
    bad = tmp_path / (
        f"Garage 61 - X - Car - Track - {int(mins):02d}.{secs:06.3f} - "
        f"01FAKEULID0000000000000002.csv"
    )
    truncated.to_csv(bad, index=False)

    with pytest.raises(ValueError, match="does not match"):
        read_garage61_csv(bad)


def test_laps_with_differing_length_estimates_align(g61_csv, tmp_path, ibt_path) -> None:
    """Two CSV laps reconstruct slightly different track lengths (different
    driven lines). Alignment must rescale to a common length, or the delta
    at S/F drifts by the length mismatch."""
    import pandas as pd

    from iracing_core import align_laps, delta_time
    from iracing_core.garage61 import read_garage61_csv

    path, original = g61_csv
    lap_a = read_garage61_csv(path)

    # same lap, but speeds scaled +0.5% => +0.5% longer reconstructed track
    frame = pd.read_csv(path)
    frame["Speed"] = frame["Speed"] * 1.005
    other = tmp_path / path.name
    frame.to_csv(other, index=False)
    lap_b = read_garage61_csv(other)
    assert lap_b.meta.track_length_m == pytest.approx(
        lap_a.meta.track_length_m * 1.005, rel=1e-4
    )

    aligned = align_laps([lap_a, lap_b])
    d = delta_time(aligned.laps[1], aligned.laps[0])
    expected = lap_b.lap_time - lap_a.lap_time  # ~0 (same filename time)
    assert d[-1] == pytest.approx(expected, abs=0.02)


def test_import_into_library(g61_csv, tmp_path) -> None:
    from iracing_core.garage61 import import_garage61_csv
    from iracing_core.store import LapLibrary

    path, original = g61_csv
    library = LapLibrary(tmp_path / "library")
    rec = import_garage61_csv(library, path, tags=("garage61",), is_reference=True)

    assert rec is not None
    assert rec.driver_name == "Test Driver"
    assert rec.is_reference
    assert "garage61" in rec.tags
    assert rec.lap_time == pytest.approx(original.lap_time, abs=0.02)

    # dedupe on re-import
    again = import_garage61_csv(library, path)
    assert again.lap_id == rec.lap_id
    assert len(library.list_laps()) == 1

    # round-trip out of the library
    stored = library.get_lap(rec.lap_id)
    np.testing.assert_allclose(stored.channel("Speed"), original.channel("Speed"), atol=1e-4)


def test_filename_time_roundtrip_leading_zeros_and_millis() -> None:
    """format -> parse must round-trip exactly, including leading-zero fields
    (single-digit minutes/seconds, zero ms, zero seconds) and the carry-over
    case where seconds would round up past 59 (the ``02.60.000`` trap)."""
    from iracing_core.garage61 import format_garage61_filename, parse_garage61_filename

    base_id = "01KXG5HFAB6EPNJ7BMMFRV13JB"  # 26-char, ULID-charset valid
    cases = [
        # (lap_time_s, expected MM.SS.mmm, lap_id)
        (0.005, "00.00.005", "02KXG5HFAB6EPNJ7BMMFRV13JB"),   # all leading zeros
        (59.999, "00.59.999", "03KXG5HFAB6EPNJ7BMMFRV13JB"),   # sub-minute, leading 00 min
        (60.0, "01.00.000", "04KXG5HFAB6EPNJ7BMMFRV13JB"),     # exact minute, zero sec/ms
        (74.123, "01.14.123", "05KXG5HFAB6EPNJ7BMMFRV13JB"),
        (123.004, "02.03.004", "06KXG5HFAB6EPNJ7BMMFRV13JB"),  # seconds field has leading 0
    ]
    for lap_time, time_txt, lap_id in cases:
        name = format_garage61_filename("Driver", "Car", "Track", lap_time, lap_id)
        # exact composed string with the expected (leading-zero) time field
        assert name == f"Garage 61 - Driver - Car - Track - {time_txt} - {lap_id}.csv"

        info = parse_garage61_filename(name)
        assert info["driver"] == "Driver"
        assert info["car"] == "Car"
        assert info["track"] == "Track"
        assert info["lap_id"] == lap_id
        assert info["lap_time"] == pytest.approx(lap_time, abs=5e-4)

        # parse -> format round-trips to the identical filename
        assert (
            format_garage61_filename(
                info["driver"], info["car"], info["track"],
                info["lap_time"], info["lap_id"],
            )
            == name
        )

    # seconds rounding up must carry into the minute field, never emit "..60.."
    carry = format_garage61_filename("D", "C", "T", 119.9996, base_id)
    assert "02.00.000" in carry and "60" not in carry.split(" - ")[-2]
    assert parse_garage61_filename(carry)["lap_time"] == pytest.approx(120.0, abs=5e-4)


def test_official_time_shorter_than_telemetry_is_rejected(g61_csv, tmp_path) -> None:
    """A filename claiming a lap time *shorter* than the recorded telemetry
    duration leaves a negative time budget for the missing distance —
    physically impossible, so the importer must reject it. This exercises the
    negative-remaining-time branch, complementing the zero-remaining case in
    test_mismatched_telemetry_is_rejected."""
    import pandas as pd

    from iracing_core.garage61 import read_garage61_csv

    path, _ = g61_csv
    frame = pd.read_csv(path)
    cut = int(len(frame) * 0.96)
    truncated = frame.iloc[:cut]

    # filename claims a lap time 5 s shorter than the data we actually hold
    fake_time = cut / 60.0 - 5.0
    mins, secs = divmod(fake_time, 60.0)
    bad = tmp_path / (
        f"Garage 61 - X - Car - Track - {int(mins):02d}.{secs:06.3f} - "
        f"07KXG5HFAB6EPNJ7BMMFRV13JB.csv"
    )
    truncated.to_csv(bad, index=False)

    with pytest.raises(ValueError, match="does not match"):
        read_garage61_csv(bad)


def test_non_ascii_metadata_survives_roundtrip(g61_csv, tmp_path) -> None:
    """Driver/car/track names containing ö, é, ń must survive the full
    format -> write -> parse -> read pipeline unchanged."""
    from iracing_core.garage61 import (
        format_garage61_filename,
        parse_garage61_filename,
        read_garage61_csv,
    )

    _, original = g61_csv
    driver = "Köster"              # ö
    car = "Café GT3"              # é
    track = "Circuit de Gdańsk"   # ń
    lap_id = "08KXG5HFAB6EPNJ7BMMFRV13JB"

    name = format_garage61_filename(driver, car, track, original.lap_time, lap_id)

    # filename-level round trip keeps every non-ASCII character intact
    info = parse_garage61_filename(name)
    assert info["driver"] == driver
    assert info["car"] == car
    assert info["track"] == track
    assert info["lap_id"] == lap_id
    assert (
        format_garage61_filename(
            info["driver"], info["car"], info["track"],
            info["lap_time"], info["lap_id"],
        )
        == name
    )

    # and it survives an actual CSV read, flowing into the resulting LapData
    path = _write_g61_csv(tmp_path / name, original, original.lap_time)
    lap = read_garage61_csv(path)
    assert lap.meta.driver_name == driver
    assert lap.meta.car_screen_name == car
    assert lap.meta.track_display_name == track
    assert lap.lap_time == pytest.approx(original.lap_time, abs=0.02)
