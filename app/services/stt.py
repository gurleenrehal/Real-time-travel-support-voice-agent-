"""
Speech-to-text service.

Real path: faster-whisper, if the package is installed AND a model can
be loaded (the model weights are downloaded from Hugging Face on first
use, which requires network access to huggingface.co -- unavailable in
this project's development sandbox, and not guaranteed in every
deployment environment either). We treat "package installed and model
loads" as the gate, and fail over cleanly if not.

Mock path (default in this repo, and what the test suite exercises):
deterministic and offline. It does NOT synthesize a fake transcript from
nothing -- the frontend and test harness send the *ground-truth spoken
text* UTF-8-encoded as the "audio" payload, and the mock STT decodes it
back out. This is explicitly a stand-in for a real audio codec path, not
a claim that transcription is happening. See docs/ARCHITECTURE.md for
exactly what would need to change to make this a real audio pipeline.
"""
from __future__ import annotations

from app.config import get_settings

try:
    from faster_whisper import WhisperModel  # type: ignore
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False


class STTService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        if _FASTER_WHISPER_AVAILABLE:
            try:
                self._model = WhisperModel(self.settings.whisper_model_size, device="cpu")
            except Exception:
                self._model = None  # model weights unavailable; fall back to mock

    @property
    def is_mock_mode(self) -> bool:
        return self._model is None

    def transcribe(self, audio_bytes: bytes) -> str:
        if self._model is not None:
            segments, _info = self._model.transcribe(audio_bytes)
            return " ".join(seg.text.strip() for seg in segments)
        return self._transcribe_mock(audio_bytes)

    @staticmethod
    def _transcribe_mock(audio_bytes: bytes) -> str:
        try:
            return audio_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return "[unrecognized audio - STT running in offline mock mode]"
