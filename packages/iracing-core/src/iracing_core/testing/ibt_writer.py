"""Minimal iRacing ``.ibt`` telemetry file writer.

Produces files binary-compatible with the irsdk disk format as parsed by
pyirsdk (``irsdk.IBT`` for channel data, ``irsdk.IRSDK(test_file=...)`` for
the embedded session-info YAML). Used to build test fixtures and demo
sessions; pyirsdk itself is used as the round-trip oracle in the tests.

Layout (little-endian, matching ``irsdk.Header`` / ``DiskSubHeader`` /
``VarHeader``):

====================  ==========================================
offset 0              Header, 112 bytes (48 + 4 VarBuffer slots)
offset 112            DiskSubHeader, 32 bytes (``Q d d i i``)
offset 144            session-info YAML blob (cp1252/ASCII)
144 + yaml_len        VarHeaders, 144 bytes per channel
after var headers     data: record_count rows of buf_len bytes
====================  ==========================================
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import yaml

__all__ = ["write_ibt", "session_info_to_yaml"]

_HEADER_SIZE = 112
_DISK_SUB_HEADER_SIZE = 32
_VAR_HEADER_SIZE = 144

# irsdk_VarType indices (see irsdk.VAR_TYPE_MAP = ['c', '?', 'i', 'I', 'f', 'd'])
_TYPE_BY_KIND: dict[str, int] = {"bool": 1, "int": 2, "bitfield": 3, "float": 4, "double": 5}
_ITEMSIZE = {1: 1, 2: 4, 3: 4, 4: 4, 5: 8}
_NP_DTYPE = {1: np.dtype("?"), 2: np.dtype("<i4"), 3: np.dtype("<u4"), 4: np.dtype("<f4"), 5: np.dtype("<f8")}

#: units/descriptions for well-known channels (cosmetic, mirrors iRacing)
_CHANNEL_UNITS: dict[str, str] = {
    "SessionTime": "s",
    "SessionTick": "",
    "Lap": "",
    "LapDist": "m",
    "LapDistPct": "%",
    "Speed": "m/s",
    "Throttle": "%",
    "Brake": "%",
    "SteeringWheelAngle": "rad",
    "RPM": "revs/min",
    "Gear": "",
    "LatAccel": "m/s^2",
    "LongAccel": "m/s^2",
    "Lat": "deg",
    "Lon": "deg",
    "OnPitRoad": "",
    "PlayerTrackSurface": "irsdk_TrkLoc",
    "FuelLevel": "l",
}


def _var_type_for(arr: np.ndarray, name: str) -> int:
    if arr.dtype == np.bool_:
        return _TYPE_BY_KIND["bool"]
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        return _TYPE_BY_KIND["bitfield"]
    if np.issubdtype(arr.dtype, np.integer):
        return _TYPE_BY_KIND["int"]
    if arr.dtype == np.float64:
        return _TYPE_BY_KIND["double"]
    if np.issubdtype(arr.dtype, np.floating):
        return _TYPE_BY_KIND["float"]
    raise TypeError(f"channel {name}: unsupported dtype {arr.dtype}")


def session_info_to_yaml(session_info: dict) -> str:
    """Serialize session info the way iRacing does: ``---`` document start,
    top-level sections separated by blank lines (pyirsdk's per-section parser
    requires the ``\\n<Key>:\\n ... \\n\\n`` shape)."""
    parts = []
    for key, value in session_info.items():
        parts.append(yaml.safe_dump({key: value}, default_flow_style=False, sort_keys=False))
    return "---\n" + "\n".join(parts) + "\n"


def write_ibt(
    path: str | Path,
    channels: dict[str, np.ndarray],
    session_info: dict | str,
    tick_rate: int = 60,
    session_start_date: int = 1_752_000_000,
) -> Path:
    """Write ``channels`` (equal-length 1-D arrays) and ``session_info`` to
    a pyirsdk-compatible ``.ibt`` file. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not channels:
        raise ValueError("at least one channel required")
    arrays = {name: np.ascontiguousarray(arr) for name, arr in channels.items()}
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"channel lengths differ: {lengths}")
    n_records = lengths.pop()

    yaml_blob = session_info if isinstance(session_info, str) else session_info_to_yaml(session_info)
    yaml_bytes = yaml_blob.encode("cp1252", errors="replace")

    # --- layout ---------------------------------------------------------
    specs = []  # (name, var_type, offset_in_record, np_dtype)
    offset = 0
    for name, arr in arrays.items():
        var_type = _var_type_for(arr, name)
        specs.append((name, var_type, offset, _NP_DTYPE[var_type]))
        offset += _ITEMSIZE[var_type]
    buf_len = offset

    session_info_offset = _HEADER_SIZE + _DISK_SUB_HEADER_SIZE
    var_header_offset = session_info_offset + len(yaml_bytes)
    data_offset = var_header_offset + len(specs) * _VAR_HEADER_SIZE

    # --- header ---------------------------------------------------------
    header = struct.pack(
        "<11iB3x",
        2,  # version
        1,  # status: connected
        tick_rate,
        1,  # session_info_update
        len(yaml_bytes),
        session_info_offset,
        len(specs),  # num_vars
        var_header_offset,
        1,  # num_buf
        buf_len,
        n_records,  # cur_buf_tick_count
        0,  # cur_buf
    )
    var_buf0 = struct.pack("<3i4x", n_records, data_offset, n_records)
    var_bufs = var_buf0 + b"\x00" * 16 * 3
    assert len(header) + len(var_bufs) == _HEADER_SIZE

    lap = arrays.get("Lap")
    lap_count = int(lap.max() - lap.min()) if lap is not None and n_records else 0
    session_time = arrays.get("SessionTime")
    end_time = float(session_time[-1]) if session_time is not None and n_records else 0.0
    disk_sub = struct.pack("<Qddii", session_start_date, 0.0, end_time, lap_count, n_records)

    # --- var headers ------------------------------------------------------
    var_headers = bytearray()
    for name, var_type, ch_offset, _ in specs:
        var_headers += struct.pack(
            "<iii?3x32s64s32s",
            var_type,
            ch_offset,
            1,  # count
            False,  # count_as_time
            name.encode("ascii"),
            f"{name} (synthetic)".encode("ascii"),
            _CHANNEL_UNITS.get(name, "").encode("ascii"),
        )

    # --- data records (vectorized) ---------------------------------------
    data = np.zeros((n_records, buf_len), dtype=np.uint8)
    for name, var_type, ch_offset, np_dtype in specs:
        raw = arrays[name].astype(np_dtype, copy=False).view(np.uint8).reshape(n_records, np_dtype.itemsize)
        data[:, ch_offset : ch_offset + np_dtype.itemsize] = raw

    with open(path, "wb") as f:
        f.write(header)
        f.write(var_bufs)
        f.write(disk_sub)
        f.write(yaml_bytes)
        f.write(bytes(var_headers))
        f.write(data.tobytes())
    return path
