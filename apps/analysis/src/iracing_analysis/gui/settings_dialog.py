"""Settings dialog: analysis knobs + optional AI endpoint configuration."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..config import AppConfig, save_config

__all__ = ["SettingsDialog"]


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, config_path: str | Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config
        self.config_path = config_path

        # -- analysis ---------------------------------------------------
        self.telemetry_edit = QLineEdit(config.telemetry_dir or "")
        self.telemetry_edit.setPlaceholderText("Documents/iRacing/telemetry (default)")
        self.minisector_spin = QSpinBox()
        self.minisector_spin.setRange(2, 200)
        self.minisector_spin.setValue(config.minisector_count)

        analysis_box = QGroupBox("Analysis")
        analysis_form = QFormLayout(analysis_box)
        analysis_form.addRow("Telemetry folder:", self.telemetry_edit)
        analysis_form.addRow("Mini-sector count:", self.minisector_spin)

        # -- AI ------------------------------------------------------------
        self.ai_enabled = QCheckBox("Enable AI coaching reports (local LLM)")
        self.ai_enabled.setChecked(config.ai.enabled)
        self.base_url_edit = QLineEdit(config.ai.base_url)
        self.api_key_edit = QLineEdit(config.ai.api_key)
        self.model_edit = QLineEdit(config.ai.model)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(5.0, 3600.0)
        self.timeout_spin.setValue(config.ai.timeout_s)
        self.test_button = QPushButton("Test connection")
        self.test_button.clicked.connect(self._test_connection)
        self.test_result = QLabel("")

        ai_box = QGroupBox("AI insights (optional — OpenAI-compatible endpoint)")
        ai_form = QFormLayout(ai_box)
        ai_form.addRow(self.ai_enabled)
        ai_form.addRow("Base URL:", self.base_url_edit)
        ai_form.addRow("API key:", self.api_key_edit)
        ai_form.addRow("Model:", self.model_edit)
        ai_form.addRow("Timeout [s]:", self.timeout_spin)
        ai_form.addRow(self.test_button, self.test_result)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(analysis_box)
        layout.addWidget(ai_box)
        layout.addWidget(buttons)

    def _test_connection(self) -> None:
        from ..config import AiConfig
        from ..insights.openai_provider import OpenAICompatibleProvider

        cfg = AiConfig(
            enabled=True,
            base_url=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip() or "none",
            model=self.model_edit.text().strip(),
            timeout_s=min(10.0, self.timeout_spin.value()),
        )
        ok = OpenAICompatibleProvider(cfg).available()
        self.test_result.setText("✓ endpoint reachable" if ok else "✗ endpoint unreachable")

    def accept(self) -> None:  # noqa: D102
        self.config.telemetry_dir = self.telemetry_edit.text().strip() or None
        self.config.minisector_count = self.minisector_spin.value()
        self.config.ai.enabled = self.ai_enabled.isChecked()
        self.config.ai.base_url = self.base_url_edit.text().strip()
        self.config.ai.api_key = self.api_key_edit.text().strip() or "none"
        self.config.ai.model = self.model_edit.text().strip()
        self.config.ai.timeout_s = self.timeout_spin.value()
        save_config(self.config, self.config_path)
        super().accept()
