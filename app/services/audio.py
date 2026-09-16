"""
Audio helpers: base64 <-> bytes conversion, and the barge-in /
interruption abstraction.

LIMITATION (documented per project requirements): the mock TTS path
returns a complete WAV byte string rather than a live audio stream, so
there is nothing server-side actually "playing" that true mid-stream
cancellation could interrupt. What IS implemented and real: a
per-session `PlaybackController` that tracks whether a TTS response is
"in flight" for a session and exposes `interrupt()`, which the WebSocket
handler calls the moment a new "audio"/"text" event arrives while a
previous turn's playback flag is still set. This is the correct
abstraction boundary for barge-in; wiring it to true streaming
cancellation is a matter of the TTS provider's streaming API (ElevenLabs
supports stream cancellation via closing the HTTP stream) once real
streaming TTS is integrated.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def base64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


@dataclass
class PlaybackController:
    """Tracks in-flight TTS playback for one session, for barge-in."""

    is_playing: bool = False
    interrupted: bool = False
    _current_turn_id: str | None = field(default=None)

    def start(self, turn_id: str) -> None:
        self._current_turn_id = turn_id
        self.is_playing = True
        self.interrupted = False

    def finish(self, turn_id: str) -> None:
        if self._current_turn_id == turn_id:
            self.is_playing = False

    def interrupt(self) -> bool:
        """Called when new user input arrives. Returns True if it actually
        interrupted an in-flight playback (i.e. genuine barge-in occurred)."""
        was_playing = self.is_playing
        if was_playing:
            self.interrupted = True
            self.is_playing = False
        return was_playing
