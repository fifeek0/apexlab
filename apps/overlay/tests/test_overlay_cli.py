"""The overlay skeleton CLI proves the shared core is usable from the overlay."""

from __future__ import annotations


def test_overlay_check_runs_offline() -> None:
    from iracing_overlay.__main__ import main

    # --check must work without a sim: report core availability and exit 0
    assert main(["--check"]) == 0
