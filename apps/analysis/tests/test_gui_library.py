"""Phase 6 GUI: the reference-library tab."""

from __future__ import annotations

import pytest
from iracing_core import LapData, LapLibrary


@pytest.fixture()
def filled_library(tmp_path, ibt_path) -> LapLibrary:
    lib = LapLibrary(tmp_path / "library")
    lib.import_ibt(ibt_path, laps="clean", tags=("own",))
    return lib


def test_library_view_lists_and_emits(qtbot, filled_library) -> None:
    from iracing_analysis.gui.library_view import LibraryView

    view = LibraryView(filled_library)
    qtbot.addWidget(view)
    assert view.table.rowCount() == 3

    emitted: list[LapData] = []
    view.addToAnalysis.connect(emitted.append)
    view.table.selectRow(0)
    view._emit_selected()
    assert len(emitted) == 1
    assert isinstance(emitted[0], LapData)
    assert emitted[0].lap_time is not None


def test_library_view_imports_csv_and_ibt_paths(qtbot, tmp_path, ibt_path, clean_laps) -> None:
    """The import routine routes .ibt and Garage 61 .csv files correctly."""
    import numpy as np
    import pandas as pd

    from iracing_analysis.gui.library_view import LibraryView

    lap = clean_laps[0]
    n = lap.n_samples
    csv_path = tmp_path / (
        "Garage 61 - Ref Driver - Formula Fable GT3 - Fantasia International "
        "- 01.21.000 - 01FAKEULID0000000000000001.csv"
    )
    pd.DataFrame(
        {
            "Speed": lap.channel("Speed"),
            "LapDistPct": lap.channel("LapDistPct"),
            "Brake": lap.channel("Brake"),
            "Throttle": lap.channel("Throttle"),
            "Gear": lap.channel("Gear"),
            "SteeringWheelAngle": lap.channel("SteeringWheelAngle"),
            "RPM": np.asarray(lap.channel("RPM")),
        }
    ).to_csv(csv_path, index=False)

    lib = LapLibrary(tmp_path / "lib2")
    view = LibraryView(lib)
    qtbot.addWidget(view)
    view.import_paths([str(ibt_path), str(csv_path)])

    records = lib.list_laps()
    drivers = {r.driver_name for r in records}
    assert "Ref Driver" in drivers  # CSV route
    assert "Test Driver" in drivers  # .ibt route
    assert view.table.rowCount() == len(records)


def test_main_window_merges_library_lap(qtbot, clean_laps, filled_library, tmp_path) -> None:
    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=str(tmp_path), library=filled_library)
    qtbot.addWidget(win)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Library" in tabs  # library available before any analysis

    win.open_laps(clean_laps[:2])
    assert len(win.workspace.laps) == 2

    lib_lap = filled_library.get_lap(filled_library.list_laps()[0].lap_id)
    win.add_lap_to_analysis(lib_lap)
    assert len(win.workspace.laps) == 3
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Library" in tabs and "Traces" in tabs
