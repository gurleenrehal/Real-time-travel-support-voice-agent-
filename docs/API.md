# API Reference

## REST endpoints

### `GET /health`
Returns service status and which backends are in mock mode.
```json
{"status": "ok", "llm_mock_mode": true, "tts_mock_mode": true, "vector_store_ready": true}
```

### `GET /metrics`
Rolling summary (last 200 turns) of per-stage latency, averaged.

### `POST /chat`
Text-only turn through the full LangGraph pipeline (no audio).
```json
// request
{"session_id": "abc", "text": "What is the checked baggage weight limit?"}
// response: TurnResult (see app/schemas.py) with response_text, retrieved_documents,
// tool_calls, confidence, handoff, latency
```

### `POST /transcribe`
Runs STT only, on a base64-encoded audio payload. Returns `is_mock: true`
in this deployment unless a working faster-whisper model is configured.

### `POST /synthesize`
Runs TTS only. Returns `is_mock: true` unless `ELEVENLABS_API_KEY` is set.

## WebSocket protocol — `/ws/voice`

Every message, both directions, is JSON with a top-level `"type"`.

### Client → Server events
| type    | payload                                  | meaning                          |
|---------|-------------------------------------------|-----------------------------------|
| `start` | (unused; connection accept sends `ready`) | reserved for future explicit start handshake |
| `audio` | `{audio_base64: str}`                     | one complete audio turn (see STT limitation below) |
| `text`  | `{text: str}`                             | one complete text turn |
| `end`   | —                                          | close the session |

### Server → Client events
| type             | payload                                                        |
|------------------|-----------------------------------------------------------------|
| `ready`          | `{}` — sent once, right after connect, with the session_id |
| `transcript`     | `{text, is_mock_stt}` — only sent for `audio` turns |
| `agent_response` | `{text, confidence, used_doc_ids, barge_in_detected}` |
| `audio_response` | `{audio_base64, is_mock_tts}` |
| `metrics`        | per-stage latency dict: `{stt_ms, retrieval_ms, tool_ms, llm_ms, tts_ms, total_ms}` |
| `handoff`        | `{handoff_required, reason, priority, conversation_summary}` — only sent when true |
| `error`          | `{message}` |
| `end`            | `{}` — echoed back after a client `end` |

### Example turn (text)
```
Client -> {"type": "text", "text": "What is the checked baggage weight limit?"}
Server -> {"type": "agent_response", "payload": {"text": "Based on our travel-support policy: ...", "confidence": 0.564, ...}}
Server -> {"type": "audio_response", "payload": {"audio_base64": "...", "is_mock_tts": true}}
Server -> {"type": "metrics", "payload": {"retrieval_ms": 1.4, "tool_ms": 0.02, "llm_ms": 0.01, "tts_ms": 3.2, "total_ms": 5.1}}
```

### STT/TTS honesty note
The `audio` event's mock STT path expects the "audio" bytes to actually be
the ground-truth spoken text, UTF-8 encoded — it is a stand-in for a real
codec path, used so the protocol and graph can be tested end-to-end
without a real Whisper model or network access. Real microphone audio
recorded by the frontend demo will decode to the mock's fallback string
unless a working faster-whisper install is configured (see
docs/ARCHITECTURE.md).
