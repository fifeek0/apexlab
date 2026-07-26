"""Shared colours/formatting for lap curves across all views."""

from __future__ import annotations

from PySide6.QtGui import QColor

__all__ = ["lap_color", "format_channel", "CHANNEL_FORMATS"]

# colour-blind-friendly cycle (Okabe–Ito), reference lap first
_LAP_COLORS = [
    "#e69f00",  # orange
    "#56b4e9",  # sky blue
    "#009e73",  # bluish green
    "#f0e442",  # yellow
    "#cc79a7",  # reddish purple
    "#d55e00",  # vermillion
    "#0072b2",  # blue
    "#999999",  # grey
]


def lap_color(index: int) -> QColor:
    return QColor(_LAP_COLORS[index % len(_LAP_COLORS)])


#: label, converter (raw -> display), format string
CHANNEL_FORMATS: dict[str, tuple[str, float, str]] = {
    "Delta": ("Δt s", 1.0, "{:+.3f}"),
    "Speed": ("Speed km/h", 3.6, "{:.1f}"),
    "Throttle": ("Throttle %", 100.0, "{:.0f}"),
    "Brake": ("Brake %", 100.0, "{:.0f}"),
    "SteeringWheelAngle": ("Steering °", 57.29577951308232, "{:.1f}"),
    "RPM": ("RPM", 1.0, "{:.0f}"),
    "Gear": ("Gear", 1.0, "{:.0f}"),
    "Time": ("Elapsed s", 1.0, "{:.3f}"),
}


def format_channel(name: str, raw_value: float) -> str:
    label, factor, fmt = CHANNEL_FORMATS.get(name, (name, 1.0, "{:.2f}"))
    return fmt.format(raw_value * factor)
