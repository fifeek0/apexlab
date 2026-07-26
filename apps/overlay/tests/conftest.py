"""Overlay tests: offscreen Qt + a reference lap and a slower live stream."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest


@pytest.fixture(scope="session")
def ref_and_stream(tmp_path_factory):
    from dataclasses import replace

    from iracing_core import IbtReader, extract_laps
    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import DriverParams, build_session, default_track

    base = DriverParams()
    slow = replace(base, a_lat_max=base.a_lat_max * 0.96)
    session = build_session(
        track=default_track(), n_laps=2, seed=31, with_out_in_laps=False,
        per_lap_params=[base, slow], param_spread=0.0,
    )
    path = tmp_path_factory.mktemp("overlay") / "s.ibt"
    write_ibt(path, channels=session.channels, session_info=session.session_info)
    with IbtReader(path) as reader:
        laps = [lap for lap in extract_laps(reader) if lap.is_complete]

    ch = session.channels
    idx = np.flatnonzero(ch["Lap"] == laps[1].lap_number)
    names = list(ch)
    stream = [{n: ch[n][i] for n in names} for i in idx]
    return path, laps[0], stream
