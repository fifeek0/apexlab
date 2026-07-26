"""Bloops-style live overlay: delta bar, input trace, gear hint, brake cues."""

from __future__ import annotations

import pytest


@pytest.fixture()
def window(qtbot, ref_and_stream):
    from iracing_core.live_compare import LiveComparison

    from iracing_overlay.window import OverlayWindow

    _, reference, stream = ref_and_stream
    cmp = LiveComparison(reference)
    beeps: list[int] = []
    win = OverlayWindow(cmp, beeper=beeps.append)
    qtbot.addWidget(win)
    win.show()
    return win, cmp, stream, beeps


def test_overlay_updates_from_stream(window) -> None:
    win, cmp, stream, _ = window
    for sample in stream[: len(stream) // 2]:
        state = cmp.feed(sample)
        if state is not None:
            win.update_state(state)

    text = win.delta_bar.text()
    assert text.startswith("+") or text.startswith("-")
    assert win.gear_label.text().isdigit()
    # input trace carries both live and reference curves
    assert win.trace.live_throttle.xData is not None
    assert len(win.trace.live_throttle.xData) > 10
    assert len(win.trace.ref_throttle.xData) > 10


def test_gear_hint_colors(window) -> None:
    from iracing_core.live_compare import LiveState

    win, *_ = window
    base = dict(
        lap_dist_m=100.0, lap_pct=0.03, time_delta=0.0, speed_delta_kmh=0.0,
        my_throttle=1.0, my_brake=0.0,
        ref_throttle=1.0, ref_brake=0.0, next_braking_m=200.0, lap_number=1,
    )
    win.update_state(LiveState(my_gear=3, ref_gear=4, **base))
    assert win.gear_hint() == "up"
    win.update_state(LiveState(my_gear=4, ref_gear=3, **base))
    assert win.gear_hint() == "down"
    win.update_state(LiveState(my_gear=4, ref_gear=4, **base))
    assert win.gear_hint() == "ok"


def test_braking_beeps_rise_and_fire_once_per_zone(window) -> None:
    win, cmp, stream, beeps = window
    for sample in stream:
        state = cmp.feed(sample)
        if state is not None:
            win.update_state(state)

    assert len(beeps) >= 6  # several zones × cue steps
    # tones are members of the rising three-step scale
    assert set(beeps) <= {0, 1, 2}
    # a zone produces each step at most once (no machine-gunning):
    # count consecutive duplicates
    repeats = sum(1 for a, b in zip(beeps, beeps[1:]) if a == b)
    assert repeats == 0


def test_replay_cli_smoke(ref_and_stream, tmp_path, capsys) -> None:
    from iracing_core import LapLibrary

    from iracing_overlay.__main__ import main

    path, reference, _ = ref_and_stream
    library = LapLibrary(tmp_path / "lib")
    rec = library.import_ibt(path, laps="best", is_reference=True)[0]

    rc = main(
        [
            "--replay", str(path),
            "--library-dir", str(tmp_path / "lib"),
            "--reference-id", str(rec.lap_id),
            "--max-updates", "50",
        ]
    )
    assert rc == 0
    assert "reference" in capsys.readouterr().out.lower()
