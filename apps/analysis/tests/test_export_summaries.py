"""Exporting real-telemetry analysis summaries as fine-tuning inputs."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def telemetry_dir(tmp_path, synthetic_session):
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import build_session, default_track

    root = tmp_path / "telemetry"
    write_ibt(
        root / "gr86" / "sess_a.ibt",
        channels=synthetic_session.channels,
        session_info=synthetic_session.session_info,
    )
    # a second track, but with only ONE clean lap -> no pair, must be skipped
    lonely = build_session(
        track=default_track(), n_laps=1, seed=9,
        track_name="speedonia ring", track_display_name="Speedonia Ring", session_id=777,
    )
    write_ibt(
        root / "gr86" / "sess_b.ibt",
        channels=lonely.channels,
        session_info=lonely.session_info,
    )
    return root


def test_export_pairs_each_clean_lap_with_the_best(telemetry_dir, tmp_path) -> None:
    from iracing_analysis.export_summaries import export_summaries

    out = tmp_path / "summaries.jsonl"
    stats = export_summaries(telemetry_dirs=[telemetry_dir], library_dirs=[], out_path=out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    # fixture session has 3 clean laps -> best is the reference, 2 comparisons
    assert stats.written == len(rows) == 2
    assert stats.skipped_groups >= 1  # the single-lap track can't form a pair

    for row in rows:
        assert row["source"] == "real"
        assert row["track"] == "Fantasia International"
        assert row["summary"]["total_delta_s"] > 0  # compared vs the fastest lap
        assert row["summary"]["corners"]
        assert row["id"]


def test_export_reads_reference_library_too(tmp_path, ibt_path) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.export_summaries import export_summaries

    lib = LapLibrary(tmp_path / "library")
    lib.import_ibt(ibt_path, laps="clean")

    out = tmp_path / "sums.jsonl"
    stats = export_summaries(telemetry_dirs=[], library_dirs=[tmp_path / "library"], out_path=out)
    assert stats.written == 2  # 3 clean laps -> 2 pairs vs best


def test_export_cli(telemetry_dir, tmp_path, capsys) -> None:
    from iracing_analysis.export_summaries import main

    out = tmp_path / "cli.jsonl"
    rc = main(["--telemetry-dir", str(telemetry_dir), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "2" in capsys.readouterr().out


def test_export_skips_rows_already_in_output(telemetry_dir, tmp_path) -> None:
    from iracing_analysis.export_summaries import export_summaries

    out = tmp_path / "summaries.jsonl"
    first = export_summaries(telemetry_dirs=[telemetry_dir], library_dirs=[], out_path=out)
    again = export_summaries(telemetry_dirs=[telemetry_dir], library_dirs=[], out_path=out)
    assert first.written == 2 and again.written == 0
    assert len(out.read_text().splitlines()) == 2
