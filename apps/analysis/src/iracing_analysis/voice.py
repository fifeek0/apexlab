"""Voice output for the race engineer.

Backends, best first:

* **Piper** — local neural TTS (``pip install piper-tts``) with real Polish
  voices. Download one with::

      python -m piper.download_voices pl_PL-darkman-medium \
          --data-dir ~/.iracing_analysis/voices

  Models in that folder are picked up automatically.
* **system** — pyttsx3 (SAPI5 on Windows / NSSpeech on macOS).
* **off** — silent (updates are always printed anyway).

All speakers speak on a worker thread so the telemetry loop never blocks.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

__all__ = [
    "BaseSpeaker",
    "NullSpeaker",
    "PiperSpeaker",
    "SystemSpeaker",
    "VOICES_DIR",
    "create_speaker",
    "find_default_piper_model",
]

log = logging.getLogger(__name__)

VOICES_DIR = Path.home() / ".iracing_analysis" / "voices"


def find_default_piper_model() -> Path | None:
    """First ``*.onnx`` voice in the default voices folder, if any."""
    if not VOICES_DIR.exists():
        return None
    models = sorted(VOICES_DIR.glob("*.onnx"))
    return models[0] if models else None


def _play_wav(path: Path) -> None:
    """Play a WAV file with whatever the platform offers (blocking)."""
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        subprocess.run(["aplay", "-q", str(path)], check=False)


class BaseSpeaker:
    """Queue + worker thread; subclasses implement :meth:`_speak`."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def say(self, text: str) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        self._queue.put(text)

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            try:
                self._speak(text)
            except Exception as exc:  # voice failures must never kill the pit wall
                log.warning("voice output failed: %s", exc)

    def _speak(self, text: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class NullSpeaker(BaseSpeaker):
    def say(self, text: str) -> None:  # no thread needed
        pass

    def _speak(self, text: str) -> None:
        pass


class SystemSpeaker(BaseSpeaker):
    """pyttsx3 (SAPI5 / NSSpeech) — available almost everywhere, robotic."""

    def __init__(self) -> None:
        super().__init__()
        import pyttsx3  # raises if not installed

        self._engine = pyttsx3.init()

    def _speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


class PiperSpeaker(BaseSpeaker):
    """Local neural TTS via piper-tts; renders to WAV and plays it."""

    def __init__(
        self,
        model_path: Path,
        _loader: Callable | None = None,
        _player: Callable[[Path], None] | None = None,
        keep_wav: Path | None = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self._player = _player or _play_wav
        self._keep_wav = keep_wav
        loader = _loader or self._load_piper
        self._voice = loader(self.model_path)

    @staticmethod
    def _load_piper(model_path: Path):
        from piper import PiperVoice

        return PiperVoice.load(str(model_path))

    def _speak(self, text: str) -> None:
        if self._keep_wav is not None:
            self._keep_wav.mkdir(parents=True, exist_ok=True)
            wav_path = self._keep_wav / f"line_{abs(hash(text)) % 99999}.wav"
            self._render(text, wav_path)
            self._player(wav_path)
            return
        with tempfile.TemporaryDirectory(prefix="iracing_voice_") as tmp:
            wav_path = Path(tmp) / "line.wav"
            self._render(text, wav_path)
            self._player(wav_path)

    def _render(self, text: str, wav_path: Path) -> None:
        with wave.open(str(wav_path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)


def create_speaker(
    engine: str = "auto",
    piper_model: Path | str | None = None,
    _piper_loader: Callable | None = None,
) -> BaseSpeaker:
    """Build the best available speaker.

    ``engine``: ``"off"`` (silent), ``"piper"``, ``"system"`` (pyttsx3) or
    ``"auto"`` (piper if a model is available, then system, then off).
    """
    if engine == "off":
        return NullSpeaker()

    model = Path(piper_model) if piper_model else find_default_piper_model()

    if engine in ("piper", "auto") and model is not None and model.exists():
        try:
            return PiperSpeaker(model, _loader=_piper_loader)
        except Exception as exc:
            log.warning("Piper unavailable (%s)", exc)
            if engine == "piper":
                raise
    elif engine == "piper":
        raise FileNotFoundError(
            f"Piper voice model not found ({model or VOICES_DIR}). Download one: "
            f"python -m piper.download_voices pl_PL-darkman-medium --data-dir {VOICES_DIR}"
        )

    if engine in ("system", "auto"):
        try:
            return SystemSpeaker()
        except Exception as exc:
            log.info("system TTS unavailable (%s); voice disabled", exc)
            if engine == "system":
                raise
    return NullSpeaker()
