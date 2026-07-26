"""Diagnostic CLI: one-command validation of a real .ibt file."""

from __future__ import annotations


def test_diagnose_reports_all_sections(ibt_path, capsys) -> None:
    from iracing_core.diagnose import main

    assert main([str(ibt_path)]) == 0
    out = capsys.readouterr().out

    # metadata from the YAML
    assert "Fantasia International" in out
    assert "Formula Fable GT3" in out
    assert "Test Driver" in out
    # structure
    assert "60 Hz" in out
    assert "Channels" in out
    # laps with flags
    assert "clean" in out and "out" in out
    # delta cross-check between the two best clean laps
    assert "Delta check" in out
    assert "OK" in out


def test_diagnose_missing_file_fails(tmp_path, capsys) -> None:
    from iracing_core.diagnose import main

    assert main([str(tmp_path / "nope.ibt")]) != 0
    assert "error" in capsys.readouterr().out.lower()
