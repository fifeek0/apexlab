"""Phase 0 gate: the overlay app imports the shared core."""


def test_overlay_imports_core() -> None:
    import iracing_overlay

    assert iracing_overlay.CORE_PACKAGE == "iracing_core"

    import iracing_core

    assert iracing_overlay.core is iracing_core
