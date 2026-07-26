"""Reading iRacing ``.ibt`` telemetry files (pure Python, cross-platform).

Built on pyirsdk:

* channel structure and single reads go through ``irsdk.IBT`` (note that
  ``ibt['Speed']`` returns only the *last* sample — full arrays come from
  ``get_all``, or from the vectorized fast path below);
* the embedded session-info YAML is read via ``irsdk.IRSDK(test_file=...)``,
  which is the only working session-info accessor for disk files, with a
  plain-YAML fallback for files pyirsdk's per-section parser cannot handle.

Channel reads use a numpy fast path (one strided decode of the whole record
block instead of pyirsdk's per-sample ``struct.unpack_from`` loop); the
tests assert bit-exact equality with ``irsdk.IBT.get_all``.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import TracebackType

import irsdk
import numpy as np
import yaml

from .models import SessionMeta

__all__ = ["IbtReader", "read_session_info"]

_NP_DTYPE_BY_VAR_TYPE: dict[int, np.dtype] = {
    0: np.dtype("S1"),  # char
    1: np.dtype("?"),  # bool
    2: np.dtype("<i4"),  # int
    3: np.dtype("<u4"),  # bitfield
    4: np.dtype("<f4"),  # float
    5: np.dtype("<f8"),  # double
}

_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z]\w*):\s*$", re.MULTILINE)
_UTF8_SIGN = b"---\nWeekendInfo:\n Encoding: UTF8"


class IbtReader:
    """Context-manager reader for one ``.ibt`` file.

    >>> with IbtReader("stint.ibt") as reader:
    ...     speed = reader.get_channel("Speed")
    ...     track = reader.meta.track_display_name
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ibt = irsdk.IBT()
        self._ibt.open(str(self.path))
        self._cache: dict[str, np.ndarray] = {}
        self._session_info: dict | None = None
        self._meta: SessionMeta | None = None

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._ibt.close()
        self._cache.clear()

    def __enter__(self) -> "IbtReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- structure -------------------------------------------------------

    @property
    def record_count(self) -> int:
        return int(self._ibt._disk_header.session_record_count)

    @property
    def tick_rate(self) -> int:
        return int(self._ibt._header.tick_rate)

    @property
    def channel_names(self) -> list[str]:
        return list(self._ibt.var_headers_names or [])

    def has_channel(self, name: str) -> bool:
        return name in (self._ibt._var_headers_dict or {})

    @property
    def session_start_unix(self) -> int:
        return int(self._ibt._disk_header.session_start_date)

    # -- channel data ------------------------------------------------------

    def get_channel(self, name: str) -> np.ndarray:
        """Full 60 Hz array for one channel (shape ``(n,)``, or ``(n, count)``
        for array-valued channels)."""
        if name in self._cache:
            return self._cache[name]
        var_headers = self._ibt._var_headers_dict or {}
        if name not in var_headers:
            raise KeyError(f"channel {name!r} not present in {self.path.name}")
        var = var_headers[name]
        try:
            arr = self._read_vectorized(var)
        except Exception:
            # robust fallback: pyirsdk's reference per-sample reader
            arr = np.asarray(self._ibt.get_all(name))
        arr.setflags(write=False)
        self._cache[name] = arr
        return arr

    def get_channels(self, names: list[str]) -> dict[str, np.ndarray]:
        return {name: self.get_channel(name) for name in names}

    def _read_vectorized(self, var) -> np.ndarray:
        header = self._ibt._header
        mm = self._ibt._shared_mem
        buf_offset = header.var_buf[0].buf_offset
        buf_len = header.buf_len
        dtype = _NP_DTYPE_BY_VAR_TYPE[var.type]
        count = var.count

        n_avail = max(0, (len(mm) - buf_offset) // buf_len)
        n = min(self.record_count, n_avail)
        raw = np.frombuffer(mm, dtype=np.uint8, count=n * buf_len, offset=buf_offset)
        block = raw.reshape(n, buf_len)
        width = dtype.itemsize * count
        col = np.ascontiguousarray(block[:, var.offset : var.offset + width])
        values = col.view(dtype)
        return values.ravel().copy() if count == 1 else values.copy()

    # -- session info ------------------------------------------------------

    @property
    def session_info(self) -> dict:
        """The full embedded YAML as a dict of top-level sections."""
        if self._session_info is None:
            self._session_info = read_session_info(self.path, blob=self.session_info_text)
        return self._session_info

    @property
    def session_info_text(self) -> str:
        header = self._ibt._header
        mm = self._ibt._shared_mem
        raw = mm[header.session_info_offset : header.session_info_offset + header.session_info_len]
        encoding = "utf-8" if raw.startswith(_UTF8_SIGN) else "cp1252"
        return raw.rstrip(b"\x00").decode(encoding, errors="replace")

    @property
    def meta(self) -> SessionMeta:
        if self._meta is None:
            self._meta = SessionMeta.from_session_info(
                self.path,
                self.session_info,
                tick_rate=self.tick_rate,
                record_count=self.record_count,
                session_start_unix=self.session_start_unix or None,
            )
        return self._meta


def read_session_info(path: str | Path, blob: str | None = None) -> dict:
    """Read the embedded session YAML of a ``.ibt`` file as a dict.

    Primary path: ``irsdk.IRSDK(test_file=...)`` per top-level section
    (pyirsdk applies escaping workarounds for iRacing's non-standard YAML).
    Sections it cannot return are filled from a plain ``yaml.safe_load`` of
    the whole blob.
    """
    path = Path(path)
    if blob is None:
        ibt = irsdk.IBT()
        ibt.open(str(path))
        try:
            header = ibt._header
            raw = ibt._shared_mem[
                header.session_info_offset : header.session_info_offset + header.session_info_len
            ]
            encoding = "utf-8" if raw.startswith(_UTF8_SIGN) else "cp1252"
            blob = raw.rstrip(b"\x00").decode(encoding, errors="replace")
        finally:
            ibt.close()

    keys = _TOP_LEVEL_KEY_RE.findall(blob)
    info: dict = {}

    ir = irsdk.IRSDK()
    try:
        if ir.startup(test_file=str(path)):
            for key in keys:
                try:
                    info[key] = ir[key]
                except Exception:
                    info[key] = None
    finally:
        ir.shutdown()

    missing = [k for k in keys if info.get(k) is None]
    if missing:
        try:
            parsed = yaml.safe_load(blob) or {}
            for key in missing:
                if isinstance(parsed, dict) and parsed.get(key) is not None:
                    info[key] = parsed[key]
        except yaml.YAMLError:
            pass

    return {k: v for k, v in info.items() if v is not None}
