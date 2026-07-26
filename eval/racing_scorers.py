"""Racing-coach grounding scorers — pure stdlib.

Each scorer returns (pass: bool, reason: str). They validate that the model's
response is grounded in the input summary JSON and follows the task contract,
without needing a reference answer.
"""

from __future__ import annotations

import re

__all__ = [
    "check_corner_grounding",
    "check_tyre_gate",
    "check_language",
    "check_number_grounding",
    "check_radio_length",
    "score_all",
]

_CORNER_RE = re.compile(r"\bT(\d+)\b")
_NUMBER_RE = re.compile(r"\d+\.?\d*")

_PL_MARKERS = {"się", "jest", "nie", "na", "że", "jak", "lub", "już", "też",
               "okrążenie", "zakręt", "hamowanie", "gaz", "prędkość", "strata"}
_EN_MARKERS = {"the", "you", "your", "and", "this", "that", "brake", "throttle",
               "corner", "speed", "loss", "focus", "lap"}

_TYRE_WORDS_PL = {"opon", "ciśnien", "temperatur", "kół", "tyre", "tire", "pressure"}
_TYRE_WORDS_EN = {"tyre", "tire", "pressure", "cold", "overheated", "temperature"}
_TYRE_EXCEPTION = {"brak danych", "unavailable", "not available", "no tyre data"}


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text) if len(m) <= 10]


def _allowed_numbers(summary: dict, tolerance_int: float = 0.75,
                     tolerance_dec: float = 0.06) -> set[float]:
    """All numbers that could legitimately appear in a grounded response."""
    raw: list[float] = []

    def _walk(obj):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            raw.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(summary)

    allowed: set[float] = set()
    for v in raw:
        allowed.add(v)
        allowed.add(round(v))
        allowed.add(round(v, 1))
        allowed.add(round(v, 2))
        allowed.add(abs(v))
        allowed.add(round(abs(v)))
        allowed.add(round(abs(v), 1))

    # common derived values: differences between pairs
    corners = summary.get("corners") or []
    for c in corners:
        ms = c.get("min_speed_kmh")
        ref_ms = c.get("ref_min_speed_kmh")
        if ms is not None and ref_ms is not None:
            diff = abs(ref_ms - ms)
            allowed.add(round(diff, 1))
            allowed.add(round(diff))

    lap_a = (summary.get("analysed_lap") or {}).get("lap_time_s")
    lap_r = (summary.get("reference_lap") or {}).get("lap_time_s")
    if lap_a is not None and lap_r is not None:
        allowed.add(round(abs(lap_a - lap_r), 1))
        allowed.add(round(abs(lap_a - lap_r), 2))
        allowed.add(round(abs(lap_a - lap_r), 3))

    return allowed


def check_corner_grounding(response: str, summary: dict) -> tuple[bool, str]:
    valid = {c["corner"] for c in (summary.get("corners") or [])}
    mentioned = {f"T{m}" for m in _CORNER_RE.findall(response)}
    bad = mentioned - valid
    if bad:
        return False, f"unknown corners: {sorted(bad)} (valid: {sorted(valid)})"
    return True, ""


def check_tyre_gate(response: str, summary: dict) -> tuple[bool, str]:
    available = (summary.get("signals") or {}).get("tyre_data_available", False)
    if available:
        return True, ""
    low = response.lower()
    if any(exc in low for exc in _TYRE_EXCEPTION):
        return True, ""
    words = _TYRE_WORDS_PL | _TYRE_WORDS_EN
    if any(w in low for w in words):
        return False, "tyre advice given without tyre data"
    return True, ""


def check_language(response: str, expected_lang: str) -> tuple[bool, str]:
    low = response.lower()
    words = set(re.findall(r"\b\w+\b", low))
    pl_hits = len(words & _PL_MARKERS)
    en_hits = len(words & _EN_MARKERS)
    if expected_lang == "pl":
        if en_hits > pl_hits + 2:
            return False, f"expected Polish, got mostly English (en={en_hits}, pl={pl_hits})"
    else:
        if pl_hits > en_hits + 2:
            return False, f"expected English, got mostly Polish (pl={pl_hits}, en={en_hits})"
    return True, ""


def check_number_grounding(response: str, summary: dict,
                           max_unsupported_ratio: float = 0.25) -> tuple[bool, str]:
    found = _extract_numbers(response)
    if not found:
        return True, ""
    allowed = _allowed_numbers(summary)
    unsupported = []
    for num in found:
        if num < 1.0:
            continue  # fractions, percentages — too noisy
        close = any(abs(num - a) <= (0.75 if num == round(num) else 0.06) for a in allowed)
        if not close:
            unsupported.append(num)
    ratio = len(unsupported) / len(found) if found else 0.0
    if ratio > max_unsupported_ratio:
        return False, f"{len(unsupported)}/{len(found)} numbers unsupported: {unsupported[:5]}"
    return True, ""


def check_radio_length(response: str, task: str, max_words: int = 30) -> tuple[bool, str]:
    if task != "radio":
        return True, ""
    n = len(response.split())
    if n > max_words:
        return False, f"radio response too long: {n} words (max {max_words})"
    return True, ""


def score_all(response: str, summary: dict, task: str, language: str) -> dict:
    """Run all grounding checks; return per-check results and an overall pass."""
    checks = {
        "corner_grounding": check_corner_grounding(response, summary),
        "tyre_gate": check_tyre_gate(response, summary),
        "language": check_language(response, language),
        "number_grounding": check_number_grounding(response, summary),
        "radio_length": check_radio_length(response, task),
    }
    return {
        "checks": {k: {"pass": v[0], "reason": v[1]} for k, v in checks.items()},
        "grounding_pass": all(v[0] for v in checks.values()),
        "n_pass": sum(1 for v in checks.values() if v[0]),
        "n_total": len(checks),
    }
