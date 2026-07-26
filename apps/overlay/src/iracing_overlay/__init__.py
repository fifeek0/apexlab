"""iracing_overlay — real-time in-sim coaching overlay (integration skeleton).

The original overlay codebase was not present on this machine, so this
package is the monorepo integration point for it: it depends on the shared
:mod:`iracing_core` package (telemetry ingest, .ibt import, lap store,
distance alignment + delta) exactly as the analysis app does. Drop the
existing overlay modules (cue logic, overlay UI) into this package and
replace any duplicated ingest/lap-store/alignment code with imports from
``iracing_core``.
"""

import iracing_core as core

CORE_PACKAGE = core.__name__
__version__ = "0.1.0"

__all__ = ["core", "CORE_PACKAGE", "__version__"]
