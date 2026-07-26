"""Phase 6 gate (core part): the personal reference-lap library."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def library(tmp_path):
    from iracing_core.store import LapLibrary

    return LapLibrary(tmp_path / "library")


def test_import_best_lap(library, ibt_path) -> None:
    records = library.import_ibt(ibt_path, laps="best", tags=("baseline",), is_reference=True)
    assert len(records) == 1
    rec = records[0]
    assert rec.track_display_name == "Fantasia International"
    assert rec.car_screen_name == "Formula Fable GT3"
    assert rec.driver_name == "Test Driver"
    assert rec.is_reference
    assert "baseline" in rec.tags
    assert rec.lap_time is not None and 60 < rec.lap_time < 180


def test_import_clean_laps_and_filtering(library, ibt_path) -> None:
    records = library.import_ibt(ibt_path, laps="clean", tags=("practice",))
    assert len(records) == 3  # 3 clean flying laps in the fixture

    assert len(library.list_laps(track="fantasia full")) == 3
    assert len(library.list_laps(track="nonexistent track")) == 0
    assert len(library.list_laps(tags=("practice",))) == 3
    assert len(library.list_laps(tags=("no-such-tag",))) == 0


def test_reimport_does_not_duplicate(library, ibt_path) -> None:
    first = library.import_ibt(ibt_path, laps="best")
    second = library.import_ibt(ibt_path, laps="best")
    assert len(library.list_laps()) == 1
    assert first[0].lap_id == second[0].lap_id


def test_roundtrip_lap_data(library, ibt_path, synthetic_session) -> None:
    """A lap loaded back from the library is a full LapData: channels intact
    and alignable against freshly parsed laps."""
    from iracing_core import IbtReader, align_laps, delta_time, extract_laps

    rec = library.import_ibt(ibt_path, laps="best", is_reference=True)[0]
    stored = library.get_lap(rec.lap_id)

    with IbtReader(ibt_path) as reader:
        fresh = [lap for lap in extract_laps(reader) if lap.is_clean]
    original = min(fresh, key=lambda lap: lap.lap_time)

    assert stored.lap_time == pytest.approx(original.lap_time, abs=1e-6)
    np.testing.assert_allclose(stored.channel("Speed"), original.channel("Speed"))

    # cross-source alignment: library lap as reference for a session lap
    other = max(fresh, key=lambda lap: lap.lap_time)
    aligned = align_laps([stored, other])
    d = delta_time(aligned.laps[1], aligned.laps[0])
    assert d[-1] == pytest.approx(other.lap_time - stored.lap_time, abs=0.003)


def test_tags_and_reference_flag_updates(library, ibt_path) -> None:
    rec = library.import_ibt(ibt_path, laps="best")[0]
    library.set_tags(rec.lap_id, ("pro", "dry"))
    library.set_reference(rec.lap_id, True)
    updated = library.list_laps(reference_only=True)
    assert len(updated) == 1
    assert set(updated[0].tags) == {"pro", "dry"}

    library.delete_lap(rec.lap_id)
    assert library.list_laps() == []


def test_core_public_api_surface() -> None:
    """Full shared-core surface promised in Phase 0."""
    import iracing_core

    for name in ("IbtReader", "LapData", "SessionMeta", "align_laps", "LapLibrary", "LiveTelemetry"):
        assert hasattr(iracing_core, name), f"iracing_core must export {name}"
