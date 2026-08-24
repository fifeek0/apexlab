"""Tests for --telemetry-dir validation in the analysis CLIs."""

from __future__ import annotations


def test_main_cli_missing_telemetry_dir(tmp_path, capsys) -> None:
    """A non-existent --telemetry-dir exits with rc=1 and a clean message."""
    from iracing_analysis.__main__ import main

    rc = main(["--telemetry-dir", str(tmp_path / "does_not_exist")])

    assert rc != 0
    captured = capsys.readouterr()
    assert "telemetry directory not found" in captured.err
    assert "Traceback" not in captured.out + captured.err


def test_main_cli_dir_without_ibt_files(tmp_path, capsys) -> None:
    """A directory with no .ibt files exits with rc=1 and a clean message."""
    from iracing_analysis.__main__ import main

    empty = tmp_path / "empty_telemetry"
    empty.mkdir()
    (empty / "notes.txt").write_text("no telemetry here", encoding="utf-8")

    rc = main(["--telemetry-dir", str(empty)])

    assert rc != 0
    captured = capsys.readouterr()
    assert "no .ibt files" in captured.err
    assert "Traceback" not in captured.out + captured.err
