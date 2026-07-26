"""Phase 0 gate: the analysis app imports and builds on the shared core."""


def test_analysis_imports_core() -> None:
    import iracing_analysis

    assert iracing_analysis.CORE_PACKAGE == "iracing_core"

    # The analysis app must reuse the shared core, not duplicate it.
    import iracing_core

    assert iracing_analysis.core is iracing_core
