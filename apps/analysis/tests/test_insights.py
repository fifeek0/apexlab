"""Phase 7 gate: pluggable AI insights.

With an (OpenAI-compatible) endpoint up → coherent report from the JSON
summary. With it off/unreachable → graceful degradation, app unchanged.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


# --------------------------------------------------------------------------
# fake OpenAI-compatible server
# --------------------------------------------------------------------------


class _FakeLLMHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []

    def do_POST(self):  # noqa: N802
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).captured.append(body)
        response = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": body.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "COACHING REPORT: You lose most time in T3 — brake later.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def fake_llm():
    _FakeLLMHandler.captured = []
    server = HTTPServer(("127.0.0.1", 0), _FakeLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _FakeLLMHandler
    server.shutdown()


# --------------------------------------------------------------------------
# structured summary
# --------------------------------------------------------------------------


def test_summary_is_json_serializable_and_complete(workspace) -> None:
    from iracing_analysis.insights.summary import build_summary

    summary = build_summary(workspace, lap_index=1)
    text = json.dumps(summary)  # must not raise
    assert len(text) > 200

    assert summary["session"]["track"] == "Fantasia International"
    assert summary["analysed_lap"]["lap_time_s"] > 60
    assert summary["reference_lap"]["lap_time_s"] > 60
    # the coached driver is the analysed lap's driver, never the reference's
    assert summary["analysed_lap"]["driver"] == "Test Driver"
    assert summary["session"]["driver"] == summary["analysed_lap"]["driver"]
    assert len(summary["corners"]) >= 8
    worst = summary["corners"][0]
    assert {"corner", "time_lost_s", "entry_delta_s", "exit_delta_s", "why"} <= set(worst)
    assert summary["theoretical_best"]["gap_to_actual_best_s"] >= 0
    assert summary["sectors"]["times_s"]


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def test_null_provider_is_default_and_graceful(workspace) -> None:
    from iracing_analysis.config import AiConfig
    from iracing_analysis.insights import create_provider
    from iracing_analysis.insights.base import NullProvider

    provider = create_provider(AiConfig(enabled=False))
    assert isinstance(provider, NullProvider)
    result = provider.generate({"anything": 1})
    assert not result.ok
    assert "disabled" in result.text.lower() or "disabled" in (result.error or "").lower()


def test_openai_provider_with_fake_endpoint(workspace, fake_llm) -> None:
    from iracing_analysis.config import AiConfig
    from iracing_analysis.insights import create_provider
    from iracing_analysis.insights.summary import build_summary

    base_url, handler = fake_llm
    cfg = AiConfig(enabled=True, base_url=base_url, api_key="none", model="test-gemma", timeout_s=10)
    provider = create_provider(cfg)

    summary = build_summary(workspace, lap_index=1)
    result = provider.generate(summary)
    assert result.ok, result.error
    assert "COACHING REPORT" in result.text

    request = handler.captured[0]
    assert request["model"] == "test-gemma"
    # the structured JSON summary must actually reach the model
    user_content = next(m["content"] for m in request["messages"] if m["role"] == "user")
    assert "Fantasia International" in user_content
    assert "corners" in user_content


def test_openai_provider_unreachable_degrades(workspace) -> None:
    from iracing_analysis.config import AiConfig
    from iracing_analysis.insights import create_provider

    cfg = AiConfig(
        enabled=True, base_url="http://127.0.0.1:9/v1", api_key="none", model="x", timeout_s=0.5
    )
    provider = create_provider(cfg)
    result = provider.generate({"a": 1})  # must not raise
    assert not result.ok
    assert result.error


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


def test_insights_view_generates_report(qtbot, workspace, fake_llm) -> None:
    from iracing_analysis.config import AiConfig
    from iracing_analysis.gui.insights_view import InsightsView

    base_url, _ = fake_llm
    cfg = AiConfig(enabled=True, base_url=base_url, model="test-gemma", timeout_s=10)
    view = InsightsView(workspace, cfg)
    qtbot.addWidget(view)
    view._generate_sync()
    assert "COACHING REPORT" in view.report_edit.toPlainText()


def test_insights_view_disabled_message(qtbot, workspace) -> None:
    from iracing_analysis.config import AiConfig
    from iracing_analysis.gui.insights_view import InsightsView

    view = InsightsView(workspace, AiConfig(enabled=False))
    qtbot.addWidget(view)
    view._generate_sync()
    assert "disabled" in view.report_edit.toPlainText().lower()


def test_main_window_has_report_tab(qtbot, clean_laps, tmp_path) -> None:
    from iracing_core import LapLibrary

    from iracing_analysis.gui.main_window import MainWindow

    win = MainWindow(telemetry_dir=str(tmp_path), library=LapLibrary(tmp_path / "lib"))
    qtbot.addWidget(win)
    win.open_laps(clean_laps)
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "AI Report" in tabs
