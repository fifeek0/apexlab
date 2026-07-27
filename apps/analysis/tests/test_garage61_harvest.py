"""garage61-harvest: API client, lap selection and end-to-end harvesting."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

# ---------------------------------------------------------------------------
# fake Garage 61 API
# ---------------------------------------------------------------------------

CARS = [
    {"id": 153, "name": "Toyota GR86", "platform": "iracing", "platform_id": "gr86"},
    {"id": 201, "name": "Dallara IR-01", "platform": "iracing", "platform_id": "ir01"},
]
TRACKS = [
    {"id": 444, "name": "Circuit de Spa-Francorchamps", "variant": "Grand Prix Pits", "platform": "iracing"},
    {"id": 500, "name": "Daytona International Speedway", "variant": "Oval", "platform": "iracing"},
]
CAR_GROUPS = [
    {"id": 4, "name": "Sports Car", "cars": [153], "platform": "iracing"},
    {"id": 9, "name": "Dirt Oval", "cars": [999], "platform": "iracing"},
]


def _lap(i, lap_time, driver="Driver X", **flags):
    base = {
        "id": f"01TESTAA{i:018d}",
        "lapTime": lap_time,
        "lapNumber": 3,
        "driver": {
            "firstName": driver.split()[0],
            "lastName": " ".join(driver.split()[1:]),
            "slug": driver.lower().replace(" ", "-"),
            "id": f"user-{i}",
        },
        "car": {"id": 153, "name": "Toyota GR86"},
        "track": {"id": 444, "name": "Circuit de Spa-Francorchamps", "variant": "Grand Prix Pits"},
        "clean": True,
        "incomplete": False,
        "missing": False,
        "discontinuity": False,
        "canViewTelemetry": True,
        "driverRating": 3000 + i * 100,
        "startTime": "2026-07-18T10:00:00Z",
    }
    base.update(flags)
    return base


class _FakeG61Handler(BaseHTTPRequestHandler):
    laps: list[dict] = []
    csv_bytes: bytes = b""
    requests_seen: list[str] = []
    fail_first_laps_call: bool = False
    _failed = False

    def do_GET(self):  # noqa: N802
        type(self).requests_seen.append(self.path)
        parsed = urlparse(self.path)
        if self.headers.get("Authorization") != "Bearer TESTTOKEN":
            self._send(401, {"error": "unauthorized"})
            return

        if parsed.path == "/api/v1/cars":
            self._send(200, {"total": len(CARS), "items": CARS})
        elif parsed.path == "/api/v1/tracks":
            self._send(200, {"total": len(TRACKS), "items": TRACKS})
        elif parsed.path == "/api/v1/car-groups":
            self._send(200, {"total": len(CAR_GROUPS), "items": CAR_GROUPS})
        elif parsed.path == "/api/v1/laps":
            if type(self).fail_first_laps_call and not type(self)._failed:
                type(self)._failed = True
                self._send(429, {"error": "slow down"})
                return
            query = parse_qs(parsed.query)
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["1000"])[0])
            page = type(self).laps[offset : offset + limit]
            self._send(200, {"total": len(type(self).laps), "items": page})
        elif parsed.path.startswith("/api/v1/laps/") and parsed.path.endswith("/csv"):
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(type(self).csv_bytes)))
            self.end_headers()
            self.wfile.write(type(self).csv_bytes)
        else:
            self._send(404, {"error": "not found"})

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_g61(clean_laps):
    import pandas as pd

    # telemetry served for every lap: a real synthetic lap in G61 CSV format
    lap = clean_laps[0]
    frame = pd.DataFrame(
        {
            "Speed": lap.channel("Speed"),
            "LapDistPct": lap.channel("LapDistPct"),
            "Lat": lap.channel("Lat"),
            "Lon": lap.channel("Lon"),
            "Brake": lap.channel("Brake"),
            "Throttle": lap.channel("Throttle"),
            "RPM": lap.channel("RPM"),
            "SteeringWheelAngle": lap.channel("SteeringWheelAngle"),
            "Gear": lap.channel("Gear"),
            "LatAccel": lap.channel("LatAccel"),
            "LongAccel": lap.channel("LongAccel"),
        }
    )
    _FakeG61Handler.csv_bytes = frame.to_csv(index=False).encode()
    _FakeG61Handler.requests_seen = []
    _FakeG61Handler.fail_first_laps_call = False
    _FakeG61Handler._failed = False
    # 6 usable laps around the real lap time + 2 that must be filtered out
    t = lap.lap_time
    _FakeG61Handler.laps = [
        _lap(1, t, driver="Alien One"),
        _lap(2, t + 1.2, driver="Fast Two"),
        _lap(3, t + 2.5, driver="Mid Three"),
        _lap(4, t + 4.0, driver="Mid Four"),
        _lap(5, t + 6.5, driver="Slow Five"),
        _lap(6, t + 9.0, driver="Slow Six"),
        _lap(7, t + 3.0, driver="Corrupt Seven", discontinuity=True),
        _lap(8, t + 3.5, driver="Hidden Eight", canViewTelemetry=False),
    ]
    server = HTTPServer(("127.0.0.1", 0), _FakeG61Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/v1", lap.lap_time
    server.shutdown()


def _client(base_url):
    from iracing_analysis.garage61_api import Garage61Client

    return Garage61Client(token="TESTTOKEN", base_url=base_url, min_interval_s=0.0)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


def test_client_lists_and_paginates(fake_g61) -> None:
    base_url, _ = fake_g61
    client = _client(base_url)
    assert [c["name"] for c in client.cars()] == ["Toyota GR86", "Dallara IR-01"]
    assert len(client.tracks()) == 2
    assert client.car_groups()[0]["name"] == "Sports Car"

    laps = client.find_laps(tracks=[444], cars=[153], page_size=3)
    assert len(laps) == 8  # 3 pages of 3+3+2
    pages = [p for p in _FakeG61Handler.requests_seen if "/laps?" in p]
    assert len(pages) == 3


def test_client_retries_on_429(fake_g61) -> None:
    base_url, _ = fake_g61
    _FakeG61Handler.fail_first_laps_call = True
    sleeps: list[float] = []
    from iracing_analysis.garage61_api import Garage61Client

    client = Garage61Client(
        token="TESTTOKEN", base_url=base_url, min_interval_s=0.0, _sleep=sleeps.append
    )
    laps = client.find_laps(tracks=[444])
    assert len(laps) == 8
    assert sleeps  # backed off at least once


def test_client_rate_limits_requests(fake_g61) -> None:
    base_url, _ = fake_g61
    sleeps: list[float] = []
    from iracing_analysis.garage61_api import Garage61Client

    client = Garage61Client(
        token="TESTTOKEN", base_url=base_url, min_interval_s=2.0, _sleep=sleeps.append
    )
    client.cars()
    client.tracks()
    assert sleeps and sleeps[0] > 0  # second call waited


# ---------------------------------------------------------------------------
# selection + naming + discipline
# ---------------------------------------------------------------------------


def test_selection_refs_plus_spread() -> None:
    from iracing_analysis.harvest import filter_usable, select_laps

    from iracing_analysis.harvest import _driver_name

    laps = _FakeG61Handler.laps or [_lap(i, 100 + i) for i in range(1, 9)]
    usable = filter_usable(laps)
    assert {_driver_name(lap) for lap in usable} & {"Corrupt Seven", "Hidden Eight"} == set()

    picked = select_laps(usable, refs=2, per_combo=4)
    assert len(picked) == 4
    assert picked[0].is_reference and picked[1].is_reference
    assert not picked[2].is_reference
    times = [p.item["lapTime"] for p in picked]
    assert times == sorted(times)


def test_selection_window_excludes_backmarkers() -> None:
    from iracing_analysis.harvest import select_laps

    laps = [_lap(1, 100.0), _lap(2, 101.0), _lap(3, 135.0)]  # 135 s = 135% of P1
    picked = select_laps(laps, refs=1, per_combo=10, window=1.2)
    assert [p.item["lapTime"] for p in picked] == [100.0, 101.0]


def test_canonical_filename_sanitizes_and_roundtrips() -> None:
    from iracing_core.garage61 import parse_garage61_filename

    from iracing_analysis.harvest import canonical_filename

    item = _lap(1, 603.456, driver="Evil - Guy / Test")
    name = canonical_filename(item)
    assert "/" not in name and name.endswith(".csv")
    info = parse_garage61_filename(name)
    assert info["lap_time"] == pytest.approx(603.456)
    assert info["lap_id"] == item["id"]
    assert info["car"] == "Toyota GR86"


def test_filename_time_rounding_edge() -> None:
    """59.9996 s must become 01.00.000, never 00.60.000."""
    from iracing_core.garage61 import format_garage61_filename, parse_garage61_filename

    name = format_garage61_filename("A", "B", "C", 59.9996, "01TESTAA000000000000000099")
    assert " - 01.00.000 - " in name
    assert parse_garage61_filename(name)["lap_time"] == pytest.approx(60.0)

    sub_minute = format_garage61_filename("A", "B", "C", 48.123, "01TESTAA000000000000000098")
    assert " - 00.48.123 - " in sub_minute


def test_discipline_inference() -> None:
    from iracing_analysis.harvest import infer_discipline

    assert infer_discipline("Sports Car", "Circuit de Spa-Francorchamps") == "road"
    assert infer_discipline("Formula Car", "Okayama") == "open_wheel"
    assert infer_discipline("Oval", "Daytona International Speedway") == "oval"
    assert infer_discipline("Dirt Oval", "Eldora Speedway") == "dirt_oval"
    assert infer_discipline("Dirt Road", "Wild West Ranch") == "dirt_road"
    assert infer_discipline(None, "Daytona International Speedway") == "oval"


# ---------------------------------------------------------------------------
# end-to-end harvest
# ---------------------------------------------------------------------------


def test_harvest_end_to_end_with_resume(fake_g61, tmp_path) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.harvest import HarvestConfig, harvest

    base_url, ref_time = fake_g61
    library = LapLibrary(tmp_path / "library")
    config = HarvestConfig(
        cache_dir=tmp_path / "cache",
        per_combo=4,
        refs=1,
        max_laps=10,
    )
    stats = harvest(
        client=_client(base_url),
        library=library,
        combos=[(444, "Circuit de Spa-Francorchamps", 153, "Toyota GR86", "Sports Car")],
        config=config,
    )
    assert stats.downloaded == 4
    assert stats.imported == 4

    records = library.list_laps()
    assert len(records) == 4
    refs = [r for r in records if r.is_reference]
    assert len(refs) == 1 and refs[0].driver_name == "Alien One"
    tags = set(records[0].tags)
    assert {"g61", "harvest", "discipline:road"} <= tags

    # resume: nothing downloaded the second time
    stats2 = harvest(
        client=_client(base_url), library=library,
        combos=[(444, "Circuit de Spa-Francorchamps", 153, "Toyota GR86", "Sports Car")],
        config=config,
    )
    assert stats2.downloaded == 0
    assert len(library.list_laps()) == 4

    # resume survives a cache-dir move: dedupe is keyed on lap ULIDs
    # recovered from the library, not on file paths
    stats3 = harvest(
        client=_client(base_url), library=library,
        combos=[(444, "Circuit de Spa-Francorchamps", 153, "Toyota GR86", "Sports Car")],
        config=HarvestConfig(cache_dir=tmp_path / "other_cache", per_combo=4, refs=1, max_laps=10),
    )
    assert stats3.downloaded == 0 and stats3.skipped == 4
    assert len(library.list_laps()) == 4



def test_harvest_marks_bad_telemetry_rejected_once(fake_g61, tmp_path) -> None:
    """A lap whose telemetry fails the physical-consistency check is recorded
    as rejected in the manifest and never re-downloaded."""
    from iracing_core import LapLibrary

    from iracing_analysis.harvest import HarvestConfig, harvest

    base_url, ref_time = fake_g61
    # the nahuel case: telemetry covers only ~60% of the track while the
    # official lap time equals the data duration -> physically impossible
    lines = _FakeG61Handler.csv_bytes.decode().splitlines()
    keep = int(len(lines) * 0.6)
    _FakeG61Handler.csv_bytes = "\n".join(lines[:keep]).encode()
    truncated_duration = (keep - 1) / 60.0
    _FakeG61Handler.laps = [
        _lap(21, truncated_duration, driver="Ghost One"),
        _lap(22, truncated_duration + 0.8, driver="Ghost Two"),
    ]

    library = LapLibrary(tmp_path / "library")
    config = HarvestConfig(cache_dir=tmp_path / "cache", per_combo=2, refs=1, max_laps=10)
    combos = [(444, "Circuit de Spa-Francorchamps", 153, "Toyota GR86", "Sports Car")]

    stats = harvest(client=_client(base_url), library=library, combos=combos, config=config)
    assert stats.rejected == 2 and stats.imported == 0
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert all(entry["status"] == "rejected" for entry in manifest.values())

    stats2 = harvest(client=_client(base_url), library=library, combos=combos, config=config)
    assert stats2.downloaded == 0 and stats2.rejected == 0  # terminal state


def test_harvest_dry_run_writes_nothing(fake_g61, tmp_path, capsys) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.harvest import HarvestConfig, harvest

    base_url, _ = fake_g61
    library = LapLibrary(tmp_path / "library")
    stats = harvest(
        client=_client(base_url),
        library=library,
        combos=[(444, "Circuit de Spa-Francorchamps", 153, "Toyota GR86", "Sports Car")],
        config=HarvestConfig(cache_dir=tmp_path / "cache", per_combo=4, dry_run=True),
    )
    assert stats.downloaded == 0
    assert stats.selected == 4
    assert library.list_laps() == []
    assert not any((tmp_path / "cache").rglob("*.csv"))
    assert "Alien One" in capsys.readouterr().out


def test_resolve_combos_by_name(fake_g61) -> None:
    from iracing_analysis.harvest import resolve_combos

    base_url, _ = fake_g61
    combos = resolve_combos(
        _client(base_url), car_terms=["gr86"], group_terms=[], track_terms=["spa"]
    )
    assert combos == [(444, "Circuit de Spa-Francorchamps (Grand Prix Pits)", 153, "Toyota GR86", None)]

    with pytest.raises(ValueError, match="no track"):
        resolve_combos(_client(base_url), car_terms=["gr86"], group_terms=[], track_terms=["monza"])


def test_token_resolution_order(tmp_path) -> None:
    from iracing_analysis.harvest import resolve_token

    token_file = tmp_path / "g61_token"
    assert resolve_token(None, {}, token_file) is None  # nothing anywhere

    token_file.write_text("  FILETOKEN123  \n")
    assert resolve_token(None, {}, token_file) == "FILETOKEN123"
    assert resolve_token(None, {"G61_TOKEN": "ENVTOKEN"}, token_file) == "ENVTOKEN"
    assert resolve_token("CLITOKEN", {"G61_TOKEN": "ENVTOKEN"}, token_file) == "CLITOKEN"


def test_discipline_road_courses_on_oval_facilities() -> None:
    """Rovals and road courses at speedway facilities are ROAD, not oval."""
    from iracing_analysis.harvest import infer_discipline

    assert infer_discipline(None, "Charlotte Motor Speedway (Roval)") == "road"
    assert infer_discipline(None, "Daytona International Speedway (Road Course)") == "road"
    assert infer_discipline(None, "World Wide Technology Raceway (Road Course)") == "road"
    assert infer_discipline(None, "Daytona International Speedway (Oval)") == "oval"


def test_discipline_rallycross() -> None:
    from iracing_analysis.harvest import infer_discipline

    assert infer_discipline("Rallycross", "Daytona International Speedway (Rallycross Long)") == "dirt_road"
    assert infer_discipline(None, "Lucas Oil Speedway (Rallycross)") == "dirt_road"
