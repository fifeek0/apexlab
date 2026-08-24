"""Shared validation for user-supplied telemetry directories.

Both CLIs (``python -m iracing_analysis`` and
``python -m iracing_analysis.export_summaries``) accept a
``--telemetry-dir`` argument. Before doing any work we validate the path so a
typo or an empty folder fails fast with a clear one-line message instead of a
traceback or a silent no-op.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["telemetry_dir_error"]


def telemetry_dir_error(directory: str | Path) -> str | None:
    """Return a one-line error message if ``directory`` is not a usable
    telemetry folder, or ``None`` when it looks fine.

    A usable folder exists, is a directory, and contains at least one
    ``.ibt`` file (searched recursively, mirroring
    :func:`iracing_core.scan_telemetry_dir`).
    """
    p = Path(directory)
    if not p.exists():
        return f"telemetry directory not found: {directory}"
    if not p.is_dir():
        return f"not a directory: {directory}"
    if not any(p.rglob("*.ibt")):
        return f"no .ibt files found in telemetry directory: {directory}"
    return None
