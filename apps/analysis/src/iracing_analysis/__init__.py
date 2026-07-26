"""iracing_analysis — post-session telemetry analysis app.

A free, self-hosted alternative to paid pro telemetry tools. All analysis
math builds on the shared :mod:`iracing_core` package (ingest, .ibt import,
lap store, distance alignment) which is also used by the real-time overlay.
"""

import iracing_core as core

CORE_PACKAGE = core.__name__
__version__ = "0.1.0"

__all__ = ["core", "CORE_PACKAGE", "__version__"]
