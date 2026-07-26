"""Insight provider plugin interface.

The AI module is strictly optional: the default :class:`NullProvider` keeps
the whole app fully functional with no LLM anywhere near it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["InsightResult", "InsightProvider", "NullProvider"]


@dataclass(frozen=True)
class InsightResult:
    """Outcome of one report generation (never raises across the boundary)."""

    ok: bool
    text: str
    provider: str
    model: str | None = None
    error: str | None = None


class InsightProvider(ABC):
    """Turns a structured analysis summary (JSON-able dict) into a
    natural-language coaching report."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, summary: dict) -> InsightResult:
        """Produce a report. Must catch its own failures and return
        ``ok=False`` instead of raising."""

    def available(self) -> bool:
        """Cheap best-effort availability check (used by the settings UI)."""
        return True


class NullProvider(InsightProvider):
    """Default no-op provider: AI insights are off."""

    name = "null"

    def generate(self, summary: dict) -> InsightResult:  # noqa: ARG002
        return InsightResult(
            ok=False,
            text=(
                "AI insights are disabled. Enable them in the settings and point "
                "the endpoint at any OpenAI-compatible server (e.g. vLLM serving "
                "Gemma on a DGX Spark) to get a natural-language coaching report. "
                "All numeric analysis in the other tabs is fully available without it."
            ),
            provider=self.name,
            error="disabled",
        )

    def available(self) -> bool:
        return False
