"""Telemetry watcher: auto-import of finished sessions into the library."""

from __future__ import annotations

from pathlib import Path

import pytest


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture()
def watch_env(tmp_path):
    from iracing_core.store import LapLibrary
    from iracing_core.watcher import TelemetryWatcher, WatcherConfig

    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    library = LapLibrary(tmp_path / "library")
    clock = FakeClock()
    config = WatcherConfig(
        telemetry_dir=telemetry_dir,
        library_dir=tmp_path / "library",
        settle_seconds=10.0,
        ensure_recording=False,
    )
    watcher = TelemetryWatcher(config, library=library, clock=clock)
    return telemetry_dir, library, watcher, clock


def _write_session(path: Path, synthetic_session) -> Path:
    from iracing_core.testing.ibt_writer import write_ibt

    return write_ibt(
        path, channels=synthetic_session.channels, session_info=synthetic_session.session_info
    )


def test_new_file_imported_after_settling(watch_env, synthetic_session) -> None:
    telemetry_dir, library, watcher, clock = watch_env

    assert watcher.scan_once() == []  # empty dir

    path = _write_session(telemetry_dir / "car" / "stint.ibt", synthetic_session)
    assert watcher.scan_once() == []  # fresh file: not settled yet

    clock.advance(11.0)
    records = watcher.scan_once()
    assert len(records) == 1  # mode 'best' -> fastest clean lap
    assert records[0].track_display_name == "Fantasia International"
    assert "auto-import" in records[0].tags
    assert library.has_source(path)

    clock.advance(11.0)
    assert watcher.scan_once() == []  # no re-import


def test_growing_file_not_imported_until_stable(watch_env, synthetic_session) -> None:
    telemetry_dir, library, watcher, clock = watch_env

    watcher.scan_once()  # watcher armed on an empty folder
    path = _write_session(telemetry_dir / "stint.ibt", synthetic_session)
    watcher.scan_once()

    clock.advance(6.0)
    with open(path, "ab") as f:  # session still recording: file grows
        f.write(b"\x00" * 128)
    assert watcher.scan_once() == []  # size changed -> settle timer restarts

    clock.advance(6.0)
    assert watcher.scan_once() == []  # only 6 s stable, needs 10

    clock.advance(5.0)
    assert len(watcher.scan_once()) == 1


def test_preexisting_files_skipped_by_default(watch_env, synthetic_session) -> None:
    telemetry_dir, library, watcher, clock = watch_env

    _write_session(telemetry_dir / "old_stint.ibt", synthetic_session)
    watcher.scan_once()  # first scan: existing files are marked as seen
    clock.advance(100.0)
    assert watcher.scan_once() == []
    assert library.list_laps() == []


def test_import_existing_backlog(tmp_path, synthetic_session) -> None:
    from iracing_core.store import LapLibrary
    from iracing_core.watcher import TelemetryWatcher, WatcherConfig

    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_session(telemetry_dir / "old_stint.ibt", synthetic_session)

    library = LapLibrary(tmp_path / "library")
    clock = FakeClock()
    watcher = TelemetryWatcher(
        WatcherConfig(
            telemetry_dir=telemetry_dir,
            library_dir=tmp_path / "library",
            settle_seconds=10.0,
            import_existing=True,
            import_mode="clean",
            ensure_recording=False,
        ),
        library=library,
        clock=clock,
    )
    watcher.scan_once()
    clock.advance(11.0)
    records = watcher.scan_once()
    assert len(records) == 3  # all clean laps

    # a fresh watcher over the same library must not import again
    watcher2 = TelemetryWatcher(
        WatcherConfig(
            telemetry_dir=telemetry_dir,
            library_dir=tmp_path / "library",
            settle_seconds=10.0,
            import_existing=True,
            ensure_recording=False,
        ),
        library=library,
        clock=clock,
    )
    watcher2.scan_once()
    clock.advance(11.0)
    assert watcher2.scan_once() == []
    assert len(library.list_laps()) == 3


def test_unparseable_file_is_skipped_once(watch_env) -> None:
    telemetry_dir, library, watcher, clock = watch_env

    bad = telemetry_dir / "corrupt.ibt"
    bad.write_bytes(b"this is not telemetry")
    watcher.scan_once()
    clock.advance(11.0)
    assert watcher.scan_once() == []  # no crash, nothing imported
    clock.advance(11.0)
    assert watcher.scan_once() == []  # and no retry loop
    assert library.list_laps() == []


def test_start_disk_recording_degrades_gracefully(ibt_path) -> None:
    from iracing_core.live import LiveTelemetry

    live = LiveTelemetry()
    assert live.start_disk_recording() is False  # not connected

    assert live.connect(test_file=str(ibt_path))
    try:
        # no Windows broadcast channel on this platform: False, never raises
        result = live.start_disk_recording()
        assert result in (True, False)
    finally:
        live.disconnect()


def test_cli_once_imports_backlog(tmp_path, synthetic_session, capsys) -> None:
    from iracing_core.store import LapLibrary
    from iracing_core.watcher import main

    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    _write_session(telemetry_dir / "stint.ibt", synthetic_session)

    rc = main(
        [
            "--telemetry-dir", str(telemetry_dir),
            "--library-dir", str(tmp_path / "library"),
            "--once",
            "--import-existing",
            "--settle", "0",
            "--no-recording",
        ]
    )
    assert rc == 0
    assert "imported" in capsys.readouterr().out.lower()
    assert len(LapLibrary(tmp_path / "library").list_laps()) == 1
