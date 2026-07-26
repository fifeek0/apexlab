"""Windowed entry point for the frozen distribution (no console window)."""

from __future__ import annotations

import sys

from iracing_analysis.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
