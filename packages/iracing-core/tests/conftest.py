"""Shared fixtures: one synthetic session written to a real .ibt file."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_session():
    from iracing_core.testing.synthetic import build_session, default_track

    return build_session(track=default_track(), n_laps=3, seed=42, with_out_in_laps=True)


@pytest.fixture(scope="session")
def ibt_path(tmp_path_factory, synthetic_session) -> Path:
    from iracing_core.testing.ibt_writer import write_ibt

    path = tmp_path_factory.mktemp("telemetry") / "fantasia_2026-07-17.ibt"
    write_ibt(
        path,
        channels=synthetic_session.channels,
        session_info=synthetic_session.session_info,
        tick_rate=60,
    )
    return path
