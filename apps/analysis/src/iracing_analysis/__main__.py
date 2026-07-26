"""Entry point: ``python -m iracing_analysis`` or the ``iracing-analysis`` script."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="iracing-analysis",
        description="Post-session iRacing telemetry analysis (free, self-hosted).",
    )
    parser.add_argument(
        "--telemetry-dir",
        help="folder with .ibt files (default: Documents/iRacing/telemetry or config)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="generate a synthetic demo session and open it (no iRacing needed)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from PySide6.QtWidgets import QApplication

    from .gui.main_window import MainWindow
    from .gui.theme import apply_dark_theme

    app = QApplication(sys.argv[:1])
    app.setApplicationName("iRacing Telemetry Analysis")
    apply_dark_theme(app)

    telemetry_dir = args.telemetry_dir
    if args.demo:
        telemetry_dir = _make_demo_dir()

    window = MainWindow(telemetry_dir=telemetry_dir)
    window.showMaximized()
    window.refresh_sessions()
    return app.exec()


def _make_demo_dir() -> str:
    """Write a synthetic session to a temp folder so the app can be explored
    without any iRacing telemetry at hand."""
    import tempfile
    from pathlib import Path

    from iracing_core.testing.ibt_writer import write_ibt
    from iracing_core.testing.synthetic import build_session, default_track

    demo_dir = Path(tempfile.gettempdir()) / "iracing_analysis_demo"
    demo_path = demo_dir / "formulafable" / "fantasia_demo.ibt"
    if not demo_path.exists():
        session = build_session(track=default_track(), n_laps=5, seed=3)
        write_ibt(demo_path, channels=session.channels, session_info=session.session_info)
    return str(demo_dir)


if __name__ == "__main__":
    raise SystemExit(main())
