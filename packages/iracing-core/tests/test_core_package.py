"""Phase 0 gate: the shared core package is importable."""


def test_core_imports() -> None:
    import iracing_core

    assert iracing_core.__version__
