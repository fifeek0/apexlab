"""Consistent dark theme: Qt chrome matched to the pyqtgraph plot style.

Without this the app mixes the native light toolbar/tables with dark plots,
which reads as unfinished. Applied from the entry point (not in tests).
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["apply_dark_theme"]


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    bg = QColor(24, 25, 28)
    panel = QColor(34, 36, 40)
    text = QColor(222, 224, 228)
    accent = QColor(230, 159, 0)  # matches the reference-lap colour

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, panel)
    palette.setColor(QPalette.ColorRole.AlternateBase, bg)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, panel)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 142, 148))
    disabled = QColor(110, 112, 118)
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(palette)
