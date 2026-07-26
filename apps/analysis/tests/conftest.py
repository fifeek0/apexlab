"""Fixtures for the analysis app: synthetic laps + offscreen Qt."""

from __future__ import annotations

import os

# Must be set before any Qt import: GUI tests run headless everywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def synthetic_session():
    from iracing_core.testing.synthetic import build_session, default_track

    return build_session(track=default_track(), n_laps=3, seed=42, with_out_in_laps=True)


@pytest.fixture(scope="session")
def ibt_path(tmp_path_factory, synthetic_session):
    from iracing_core.testing.ibt_writer import write_ibt

    path = tmp_path_factory.mktemp("telemetry") / "fantasia" / "fantasia_test.ibt"
    write_ibt(path, channels=synthetic_session.channels, session_info=synthetic_session.session_info)
    return path


@pytest.fixture(scope="session")
def all_laps(ibt_path):
    from iracing_core import IbtReader, extract_laps

    with IbtReader(ibt_path) as reader:
        return extract_laps(reader)


@pytest.fixture(scope="session")
def clean_laps(all_laps):
    return [lap for lap in all_laps if lap.is_clean]


@pytest.fixture()
def workspace(clean_laps):
    from iracing_analysis.analysis.workspace import Workspace

    return Workspace(clean_laps, reference_index=0)
