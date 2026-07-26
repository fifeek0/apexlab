"""Voice output backends for the race engineer (Piper / system / off)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_null_speaker_is_silent_and_safe() -> None:
    from iracing_analysis.voice import NullSpeaker, create_speaker

    speaker = create_speaker("off")
    assert isinstance(speaker, NullSpeaker)
    speaker.say("anything")  # must not raise


def test_create_speaker_auto_prefers_piper_when_model_present(tmp_path) -> None:
    from iracing_analysis.voice import PiperSpeaker, create_speaker

    model = tmp_path / "pl_PL-fake.onnx"
    model.write_bytes(b"onnx")
    speaker = create_speaker(
        "auto", piper_model=model, _piper_loader=lambda path: object()
    )
    assert isinstance(speaker, PiperSpeaker)


def test_create_speaker_auto_falls_back_when_nothing_available(tmp_path) -> None:
    from iracing_analysis.voice import NullSpeaker, create_speaker

    speaker = create_speaker("auto", piper_model=tmp_path / "missing.onnx")
    # no piper model, no pyttsx3 in this venv -> silent fallback, never an error
    assert isinstance(speaker, NullSpeaker)


def test_piper_speaker_renders_and_plays(tmp_path) -> None:
    from iracing_analysis.voice import PiperSpeaker

    synthesized: list[str] = []
    played: list[Path] = []

    class FakeVoice:
        def synthesize_wav(self, text: str, wav_file) -> None:
            synthesized.append(text)
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 100)

    speaker = PiperSpeaker(
        model_path=tmp_path / "model.onnx",
        _loader=lambda path: FakeVoice(),
        _player=played.append,
    )
    speaker._speak("Delta siedem sekund.")  # synchronous path used by the worker

    assert synthesized == ["Delta siedem sekund."]
    assert len(played) == 1


def test_default_voice_dir_discovery(tmp_path, monkeypatch) -> None:
    from iracing_analysis import voice

    monkeypatch.setattr(voice, "VOICES_DIR", tmp_path)
    assert voice.find_default_piper_model() is None
    model = tmp_path / "pl_PL-darkman-medium.onnx"
    model.write_bytes(b"x")
    assert voice.find_default_piper_model() == model


@pytest.mark.skipif(
    not (Path.home() / ".iracing_analysis" / "voices" / "pl_PL-darkman-medium.onnx").exists(),
    reason="real Piper voice model not downloaded",
)
def test_piper_real_synthesis(tmp_path) -> None:
    """Integration: the real Piper model renders a WAV for an engineer line."""
    import wave

    from iracing_analysis.voice import PiperSpeaker

    model = Path.home() / ".iracing_analysis" / "voices" / "pl_PL-darkman-medium.onnx"
    rendered: list[Path] = []
    speaker = PiperSpeaker(model_path=model, _player=rendered.append, keep_wav=tmp_path)
    speaker._speak("Okrążenie dwa czterdzieści dwa. Skup się na zakręcie osiem.")

    assert len(rendered) == 1
    with wave.open(str(rendered[0]), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    assert duration > 1.0  # an actual spoken sentence, not an empty file
