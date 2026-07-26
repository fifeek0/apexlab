"""Bloops-style in-sim overlay window.

Frameless, always-on-top, translucent strip showing, live against a
reference lap from the library:

* **delta bar** — iRacing-style: bar width scales to ±2 s, colour blends
  red→white→green with the speed difference (±5 km/h saturation), the time
  delta printed in the centre;
* **input trace** — your last ~250 m of throttle/brake as bright lines over
  the reference's inputs (shaded), including the next 150 m of the
  reference — you see the braking point *coming*;
* **gear hint** — the current gear turns green when the reference runs a
  higher gear (shift up) and red when lower;
* **audio cues** — three rising tones approaching each reference braking
  zone (inject any callable as ``beeper``; the default plays sine beeps).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Callable

import numpy as np
import pyqtgraph as pg
from iracing_core.live_compare import LiveComparison, LiveState
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

__all__ = ["OverlayWindow", "ToneBeeper"]

log = logging.getLogger(__name__)

#: cue distances before a braking zone (m) → tone step 0/1/2
_CUE_STEPS_M = (150.0, 80.0, 30.0)


class ToneBeeper:
    """Default audio backend: three short sine tones of rising pitch."""

    _FREQS = (440.0, 660.0, 880.0)

    def __init__(self) -> None:
        self._effects = []
        try:
            import tempfile
            import wave
            from pathlib import Path

            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QSoundEffect

            tone_dir = Path(tempfile.gettempdir()) / "iracing_overlay_tones"
            tone_dir.mkdir(exist_ok=True)
            for i, freq in enumerate(self._FREQS):
                path = tone_dir / f"tone{i}.wav"
                if not path.exists():
                    rate, dur = 22050, 0.09
                    n = int(rate * dur)
                    with wave.open(str(path), "wb") as f:
                        f.setnchannels(1)
                        f.setsampwidth(2)
                        f.setframerate(rate)
                        fade = np.minimum(1.0, np.linspace(0, 8, n))
                        wavef = 0.6 * np.sin(2 * math.pi * freq * np.arange(n) / rate)
                        f.writeframes((wavef * fade * 32767).astype("<i2").tobytes())
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(str(path)))
                effect.setVolume(0.7)
                self._effects.append(effect)
        except Exception as exc:  # no audio device / QtMultimedia missing
            log.info("audio cues unavailable: %s", exc)
            self._effects = []

    def __call__(self, step: int) -> None:
        if self._effects:
            self._effects[min(step, len(self._effects) - 1)].play()


class DeltaBar(QWidget):
    """iRacing-style delta bar: ±2 s width, colour from the speed gap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(46)
        self._delta = 0.0
        self._speed_delta = 0.0

    def set_values(self, time_delta: float, speed_delta_kmh: float) -> None:
        self._delta = time_delta
        self._speed_delta = speed_delta_kmh
        self.update()

    def text(self) -> str:
        return f"{self._delta:+.2f}"

    def _bar_color(self) -> QColor:
        # -5 km/h (slower) → red, 0 → white, +5 km/h (faster) → green
        f = max(-1.0, min(1.0, self._speed_delta / 5.0))
        if f >= 0:
            return QColor(int(255 * (1 - f)), 255, int(255 * (1 - f)))
        return QColor(255, int(255 * (1 + f)), int(255 * (1 + f)))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 11, 13, 215))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)

        # bar grows right when losing time (like iRacing's bar)
        half = (w - 110) / 2
        frac = max(-1.0, min(1.0, self._delta / 2.0))
        painter.setBrush(self._bar_color())
        if frac >= 0:
            painter.drawRect(int(w / 2 + 55), 8, int(half * frac), h - 16)
        else:
            painter.drawRect(int(w / 2 - 55 + half * frac), 8, int(-half * frac), h - 16)

        painter.setPen(QColor(235, 236, 240))
        font = QFont("Menlo", 17)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class InputTrace(pg.PlotWidget):
    """Your recent throttle/brake over the reference's (shaded), by distance."""

    BEHIND_M = 250.0
    AHEAD_M = 150.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground((10, 11, 13, 215))
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMouseEnabled(x=False, y=False)
        self.setXRange(-self.BEHIND_M, self.AHEAD_M, padding=0)
        self.setYRange(0, 1.05, padding=0)
        self.setMinimumHeight(120)

        self.ref_throttle = self.plot(
            [], [], pen=pg.mkPen((0, 160, 90, 160), width=1),
            fillLevel=0.0, brush=(0, 160, 90, 55),
        )
        self.ref_brake = self.plot(
            [], [], pen=pg.mkPen((200, 60, 50, 160), width=1),
            fillLevel=0.0, brush=(200, 60, 50, 55),
        )
        self.live_throttle = self.plot([], [], pen=pg.mkPen((60, 255, 140), width=2))
        self.live_brake = self.plot([], [], pen=pg.mkPen((255, 80, 70), width=2))
        self.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen("#ffffff66", width=1)))

        self._history: deque[tuple[float, float, float]] = deque(maxlen=600)

    def push_live(self, dist: float, throttle: float, brake: float) -> None:
        self._history.append((dist, throttle, brake))

    def refresh(self, current_dist: float, window) -> None:
        if window is not None:
            self.ref_throttle.setData(window.offsets_m, window.throttle)
            self.ref_brake.setData(window.offsets_m, window.brake)
        if self._history:
            data = np.asarray(self._history)
            offsets = data[:, 0] - current_dist
            keep = offsets >= -self.BEHIND_M
            self.live_throttle.setData(offsets[keep], data[keep, 1])
            self.live_brake.setData(offsets[keep], data[keep, 2])


class OverlayWindow(QWidget):
    def __init__(
        self,
        comparison: LiveComparison,
        beeper: Callable[[int], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.comparison = comparison
        self.beeper = beeper if beeper is not None else ToneBeeper()
        self._cue_step_done = -1
        self._prev_next_braking = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(520, 250)

        self.delta_bar = DeltaBar()
        self.trace = InputTrace()

        self.gear_label = QLabel("-")
        self.gear_label.setFixedSize(44, 44)
        self.gear_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_gear_style("ok")
        self.brake_label = QLabel("")
        self.brake_label.setStyleSheet(
            "color: #eceef2; background: rgba(10,11,13,215); border-radius: 6px; padding: 4px 10px;"
        )
        bottom = QHBoxLayout()
        bottom.addWidget(self.gear_label)
        bottom.addWidget(self.brake_label)
        bottom.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.delta_bar)
        layout.addWidget(self.trace)
        layout.addLayout(bottom)

        self._gear_hint = "ok"

    # -- state -----------------------------------------------------------

    def gear_hint(self) -> str:
        return self._gear_hint

    def _set_gear_style(self, hint: str) -> None:
        color = {"up": "#22a558", "down": "#c03830", "ok": "#26282e"}[hint]
        self.gear_label.setStyleSheet(
            f"color: white; background: {color}; border-radius: 22px; "
            f"font: bold 20px 'Menlo';"
        )

    def update_state(self, state: LiveState) -> None:
        self.delta_bar.set_values(state.time_delta, state.speed_delta_kmh)

        self.gear_label.setText(str(state.my_gear))
        hint = "ok"
        if state.ref_gear > state.my_gear:
            hint = "up"
        elif state.ref_gear < state.my_gear:
            hint = "down"
        if hint != self._gear_hint:
            self._gear_hint = hint
            self._set_gear_style(hint)

        # audio cues: fire each step once while closing on the zone
        nb = state.next_braking_m
        if nb is not None:
            if self._prev_next_braking is not None and nb > self._prev_next_braking + 50:
                self._cue_step_done = -1  # passed the zone; re-arm
            for step, threshold in enumerate(_CUE_STEPS_M):
                if nb <= threshold and step > self._cue_step_done:
                    self._cue_step_done = step
                    try:
                        self.beeper(step)
                    except Exception:  # audio must never kill the overlay
                        log.debug("beeper failed", exc_info=True)
            self._prev_next_braking = nb
            self.brake_label.setText(f"BRAKE in {nb:3.0f} m" if nb < 300 else "")

        self.trace.push_live(state.lap_dist_m, state.my_throttle, state.my_brake)
        self.trace.refresh(state.lap_dist_m, self.comparison.reference_window(
            behind_m=InputTrace.BEHIND_M, ahead_m=InputTrace.AHEAD_M
        ))
