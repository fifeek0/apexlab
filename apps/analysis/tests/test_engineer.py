"""RaceEngineer: per-lap 'where you lose time' radio updates vs a reference."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def ref_and_slow(tmp_path_factory):
    """Reference lap + a controlled slower lap (4% less grip)."""
    from dataclasses import replace

    from iracing_core import IbtReader, extract_laps
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import DriverParams, build_session, default_track

    base = DriverParams()
    slow = replace(base, a_lat_max=base.a_lat_max * 0.96)
    session = build_session(
        track=default_track(),
        n_laps=2,
        seed=5,
        with_out_in_laps=False,
        per_lap_params=[base, slow],
        param_spread=0.0,
    )
    path = tmp_path_factory.mktemp("engineer") / "session.ibt"
    write_ibt(path, channels=session.channels, session_info=session.session_info)
    with IbtReader(path) as reader:
        laps = [lap for lap in extract_laps(reader) if lap.is_complete]
    return laps[0], laps[1]


def test_engineer_update_structure(ref_and_slow) -> None:
    from iracing_analysis.engineer import RaceEngineer

    ref, slow = ref_and_slow
    engineer = RaceEngineer(reference=ref, language="en")
    update = engineer.on_lap(slow)

    assert update is not None
    assert update.lap_time == pytest.approx(slow.lap_time)
    assert update.delta_total == pytest.approx(slow.lap_time - ref.lap_time, abs=0.02)
    assert update.delta_total > 0
    assert 1 <= len(update.top_losses) <= 3
    worst = update.top_losses[0]
    assert worst.time_lost > 0.05
    assert worst.label.startswith("T")
    assert worst.why  # short human cue

    # the radio message mentions the delta and the worst corner
    assert worst.label in update.message
    assert f"{update.delta_total:+.2f}" in update.message


def test_engineer_polish_message(ref_and_slow) -> None:
    from iracing_analysis.engineer import RaceEngineer

    ref, slow = ref_and_slow
    update = RaceEngineer(reference=ref, language="pl").on_lap(slow)
    assert "do referencji" in update.message


def test_engineer_reports_faster_lap(ref_and_slow) -> None:
    """Reference is the slow lap -> driving the fast one reports time gained."""
    from iracing_analysis.engineer import RaceEngineer

    ref, slow = ref_and_slow
    update = RaceEngineer(reference=slow, language="en").on_lap(ref)
    assert update.delta_total < 0
    assert "faster" in update.message.lower()


def test_engineer_skips_incomplete_laps(ref_and_slow) -> None:
    from dataclasses import replace

    from iracing_analysis.engineer import RaceEngineer

    ref, slow = ref_and_slow
    partial = replace(slow, lap_time=None, is_complete=False)
    assert RaceEngineer(reference=ref).on_lap(partial) is None


def test_engineer_llm_oneliner(ref_and_slow, monkeypatch) -> None:
    """With a provider configured, the update gains a natural-language radio
    line; provider failures must never break the numeric update."""
    from iracing_analysis.engineer import RaceEngineer
    from iracing_analysis.insights.base import InsightProvider, InsightResult

    class FakeProvider(InsightProvider):
        name = "fake"

        def generate(self, summary: dict) -> InsightResult:
            assert "corners" in summary
            return InsightResult(ok=True, text="Box box... just kidding. Push in T3.", provider="fake")

    class BrokenProvider(InsightProvider):
        name = "broken"

        def generate(self, summary: dict) -> InsightResult:
            return InsightResult(ok=False, text="", provider="broken", error="down")

    ref, slow = ref_and_slow
    update = RaceEngineer(reference=ref, provider=FakeProvider()).on_lap(slow)
    assert update.radio_line == "Box box... just kidding. Push in T3."

    update = RaceEngineer(reference=ref, provider=BrokenProvider()).on_lap(slow)
    assert update is not None and update.radio_line is None
