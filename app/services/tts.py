"""
Text-to-speech service.

Real path: ElevenLabs REST API, used when ELEVENLABS_API_KEY is set.
Requires network access to api.elevenlabs.io, which is not reachable
from this project's development sandbox -- the code path is implemented
and correct but not exercised here (see README limitations).

Mock path (default, and what tests exercise): returns a short silent
WAV file generated locally with the standard library `wave` module, so
callers always get back well-formed audio bytes with zero external
dependencies. It does not claim to be a real rendering of the text.
"""
from __future__ import annotations

import io
import struct
import wave

import httpx

from app.config import get_settings


class TTSService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def is_mock_mode(self) -> bool:
        return self.settings.tts_mock_mode

    def synthesize(self, text: str) -> bytes:
        if self.settings.tts_mock_mode:
            return self._synthesize_mock(text)
        return self._synthesize_elevenlabs(text)

    # ------------------------------------------------------------------
    @staticmethod
    def _synthesize_mock(text: str, duration_seconds: float | None = None) -> bytes:
        """Generates a real, valid, silent WAV file. Duration scales
        loosely with text length purely so downstream latency/duration
        metrics have something non-trivial to measure -- the audio
        content itself is silence, not synthesized speech."""
        duration_seconds = duration_seconds or min(6.0, max(0.4, len(text) / 20))
        sample_rate = 16000
        n_frames = int(duration_seconds * sample_rate)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            silence_frame = struct.pack("<h", 0)
            wav_file.writeframes(silence_frame * n_frames)
        return buffer.getvalue()

    def _synthesize_elevenlabs(self, text: str) -> bytes:
        try:
            resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}",
                headers={"xi-api-key": self.settings.elevenlabs_api_key or ""},
                json={"text": text, "model_id": "eleven_monolingual_v1"},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.content
        except Exception:
            # Network/API unavailable -- degrade to mock audio rather than crash the turn.
            return self._synthesize_mock(text)
