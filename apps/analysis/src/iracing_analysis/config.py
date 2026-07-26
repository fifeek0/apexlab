"""App configuration: telemetry folder, analysis knobs, optional AI endpoint.

Stored as JSON in ``~/.iracing_analysis/config.json`` — human-editable and
covered by the in-app settings dialog.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["AiConfig", "AppConfig", "config_path", "load_config", "save_config"]

log = logging.getLogger(__name__)


@dataclass
class AiConfig:
    """Optional local-LLM insights endpoint (OpenAI-compatible)."""

    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "none"  # vLLM/Ollama accept any placeholder
    model: str = "google/gemma-3-27b-it"
    timeout_s: float = 120.0


@dataclass
class AppConfig:
    telemetry_dir: str | None = None  # None → iRacing default folder
    library_dir: str | None = None  # None → ~/.iracing_analysis/library
    minisector_count: int = 20
    grid_spacing_m: float = 1.0
    ai: AiConfig = field(default_factory=AiConfig)


def config_path() -> Path:
    return Path.home() / ".iracing_analysis" / "config.json"


def load_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path) if path else config_path()
    if not p.exists():
        return AppConfig()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        ai_raw = raw.pop("ai", {}) or {}
        ai = AiConfig(**{k: v for k, v in ai_raw.items() if k in AiConfig.__dataclass_fields__})
        known = {k: v for k, v in raw.items() if k in AppConfig.__dataclass_fields__ and k != "ai"}
        return AppConfig(**known, ai=ai)
    except Exception as exc:
        log.warning("bad config file %s (%s); using defaults", p, exc)
        return AppConfig()


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    p = Path(path) if path else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return p
