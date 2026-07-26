"""Insights via any OpenAI-compatible chat endpoint (vLLM, Ollama, llama.cpp).

Designed for a local/LAN LLM — e.g. Gemma served by vLLM on an NVIDIA DGX
Spark. Post-session analysis has no latency pressure, so a big local model
is the perfect fit; nothing ever leaves your network.
"""

from __future__ import annotations

import json
import logging

from ..config import AiConfig
from .base import InsightProvider, InsightResult

__all__ = ["OpenAICompatibleProvider", "SYSTEM_PROMPT", "RADIO_PROMPT"]

log = logging.getLogger(__name__)

RADIO_PROMPT = """\
You are a race engineer on the pit-wall radio. From the JSON lap summary,
say ONE short radio message to the driver (max 25 words): the delta, the one
or two corners that matter most and what to change. Speak the language given
in the 'language' field. No preamble, no markdown — just the radio line.
"""

SYSTEM_PROMPT = """\
You are a professional race engineer and driver coach analysing iRacing
telemetry. You receive a JSON summary comparing the driver's lap against a
reference lap: per-corner time deltas (entry / exit / run to the next
corner), braking points, minimum corner speeds, throttle application
points, trail-braking overlap, sector times, a theoretical-best estimate,
consistency metrics, grip usage, and (when the telemetry contains them)
tyre temperatures/pressures and track/air conditions under 'signals'.
Positive delta = time LOST to the reference. The driver being coached is
'analysed_lap.driver'; the reference belongs to another driver or lap.

Write a detailed, structured coaching report (markdown headings):

1. **Verdict** — one paragraph: where the lap time is, the dominant theme.
2. **Corner by corner** — the 4-5 corners costing the most time (use
   'time_lost_s', which includes the run to the next corner). If the
   corners list is empty (e.g. a flat-out oval lap), skip this section and
   base the analysis on sector times and the delta instead. For each:
   WHAT differs (quote the numbers: metres, km/h, seconds), WHY it costs
   time, and ONE concrete drill or cue to fix it next session.
3. **Technique themes** — read across the corners and name the recurring
   patterns explicitly, e.g.: trail-braking (too much, too little, or
   well-timed brake release into the apex), braking points (early/late),
   throttle application timing, line/steering usage. Give a cue per theme.
4. **Tyres & conditions** — ONLY if signals.tyre_data_available is true:
   comment on tyre temperatures/pressures (cold/overheated/imbalanced,
   e.g. 'fronts too cold — weave harder on the out-lap, brake earlier and
   harder for the first two corners'). If tyre data is unavailable, write
   exactly one sentence saying so and move on — never guess tyre state.
5. **Consistency** — only if more than 2 laps were compared; otherwise skip.
6. **Plan for the next session** — three prioritised items: the corner or
   habit to attack first, the drill, and what number to watch to confirm
   progress.

Be specific and quantitative. Never invent data absent from the JSON —
if a field is missing, say it is unavailable. Address the driver directly.
"""


class OpenAICompatibleProvider(InsightProvider):
    name = "openai-compatible"

    def __init__(self, config: AiConfig):
        self.config = config

    def _client(self):
        from openai import OpenAI

        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "none",
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    def available(self) -> bool:
        try:
            self._client().models.list()
            return True
        except Exception:
            return False

    def generate(self, summary: dict) -> InsightResult:
        system = RADIO_PROMPT if summary.get("task") == "radio_message" else SYSTEM_PROMPT
        try:
            response = self._client().chat.completions.create(
                model=self.config.model,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            "Telemetry analysis summary:\n\n"
                            + json.dumps(summary, indent=1)
                        ),
                    },
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return InsightResult(
                    ok=False, text="", provider=self.name, model=self.config.model,
                    error="endpoint returned an empty response",
                )
            return InsightResult(ok=True, text=text, provider=self.name, model=self.config.model)
        except Exception as exc:
            log.warning("insight generation failed: %s", exc)
            return InsightResult(
                ok=False,
                text=(
                    f"Could not reach the AI endpoint at {self.config.base_url} "
                    f"({exc.__class__.__name__}). The rest of the app is unaffected — "
                    f"check that the server is running and the settings are correct."
                ),
                provider=self.name,
                model=self.config.model,
                error=str(exc),
            )
