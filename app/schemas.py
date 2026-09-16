"""
Pydantic models for the REST API and the /ws/voice WebSocket protocol.

The WebSocket protocol is event-typed: every message, in both directions,
is a JSON object with a top-level "type" field. See docs/API.md for the
full protocol description.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Shared / domain models
# --------------------------------------------------------------------------

class RetrievedDocument(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    success: bool
    latency_ms: float


class HandoffInfo(BaseModel):
    handoff_required: bool
    reason: str | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    conversation_summary: str | None = None


class LatencyMetrics(BaseModel):
    stt_ms: float | None = None
    retrieval_ms: float | None = None
    tool_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    total_ms: float | None = None


class TurnResult(BaseModel):
    session_id: str
    turn_id: str
    user_text: str
    response_text: str
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    confidence: float
    handoff: HandoffInfo
    latency: LatencyMetrics


# --------------------------------------------------------------------------
# REST request/response models
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    text: str


class ChatResponse(BaseModel):
    turn: TurnResult


class TranscribeRequest(BaseModel):
    session_id: str
    audio_base64: str
    sample_rate: int = 16000


class TranscribeResponse(BaseModel):
    text: str
    is_mock: bool
    latency_ms: float


class SynthesizeRequest(BaseModel):
    text: str


class SynthesizeResponse(BaseModel):
    audio_base64: str
    is_mock: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_mock_mode: bool
    tts_mock_mode: bool
    vector_store_ready: bool


# --------------------------------------------------------------------------
# WebSocket protocol events
# --------------------------------------------------------------------------
# Client -> Server: start, audio, text, end
# Server -> Client: ready, transcript, agent_response, audio_response,
#                   metrics, handoff, error, end

WS_CLIENT_EVENT_TYPES = {"start", "audio", "text", "end"}
WS_SERVER_EVENT_TYPES = {
    "ready", "transcript", "agent_response", "audio_response",
    "metrics", "handoff", "error", "end",
}


class WSClientEvent(BaseModel):
    type: Literal["start", "audio", "text", "end"]
    session_id: str | None = None
    audio_base64: str | None = None
    text: str | None = None


class WSServerEvent(BaseModel):
    type: Literal[
        "ready", "transcript", "agent_response", "audio_response",
        "metrics", "handoff", "error", "end",
    ]
    session_id: str
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
