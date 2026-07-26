"""iracing-engineer CLI: replay mode drives the full pit-wall pipeline."""

from __future__ import annotations


def test_engineer_cli_replay(tmp_path, ibt_path, capsys) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.engineer import main

    library = LapLibrary(tmp_path / "library")
    ref = library.import_ibt(ibt_path, laps="best", is_reference=True)[0]

    rc = main(
        [
            "--replay", str(ibt_path),
            "--library-dir", str(tmp_path / "library"),
            "--reference-id", str(ref.lap_id),
            "--language", "en",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # updates for the flying laps (reference lap itself reports ~0.00)
    assert out.count("[lap") >= 3
    assert "to the reference" in out or "faster than the reference" in out
    assert "Focus on" in out or "Gaining in" in out


def test_engineer_cli_auto_reference(tmp_path, ibt_path, capsys) -> None:
    """Without --reference-id the fastest matching library lap is used."""
    from iracing_core import LapLibrary

    from iracing_analysis.engineer import main

    LapLibrary(tmp_path / "library").import_ibt(ibt_path, laps="clean")
    rc = main(
        [
            "--replay", str(ibt_path),
            "--library-dir", str(tmp_path / "library"),
            "--language", "pl",
        ]
    )
    assert rc == 0
    assert "do referencji" in capsys.readouterr().out
