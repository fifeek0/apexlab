"""Settings dialog: edits the config file (incl. the AI endpoint)."""

from __future__ import annotations


def test_settings_dialog_roundtrip(qtbot, tmp_path) -> None:
    from iracing_analysis.config import AppConfig, load_config
    from iracing_analysis.gui.settings_dialog import SettingsDialog

    cfg_path = tmp_path / "config.json"
    config = AppConfig(minisector_count=25)
    dialog = SettingsDialog(config, config_path=cfg_path)
    qtbot.addWidget(dialog)

    assert dialog.minisector_spin.value() == 25
    dialog.ai_enabled.setChecked(True)
    dialog.base_url_edit.setText("http://dgx-spark:8000/v1")
    dialog.model_edit.setText("google/gemma-3-27b-it")
    dialog.accept()

    saved = load_config(cfg_path)
    assert saved.ai.enabled is True
    assert saved.ai.base_url == "http://dgx-spark:8000/v1"
    assert saved.ai.model == "google/gemma-3-27b-it"
    assert saved.minisector_count == 25
