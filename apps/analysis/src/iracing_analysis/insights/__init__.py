"""Optional, pluggable AI insights (local LLM coaching reports)."""

from ..config import AiConfig
from .base import InsightProvider, InsightResult, NullProvider

__all__ = ["AiConfig", "InsightProvider", "InsightResult", "NullProvider", "create_provider"]


def create_provider(config: AiConfig) -> InsightProvider:
    """NullProvider unless AI is explicitly enabled in the config."""
    if not config.enabled:
        return NullProvider()
    from .openai_provider import OpenAICompatibleProvider

    return OpenAICompatibleProvider(config)
