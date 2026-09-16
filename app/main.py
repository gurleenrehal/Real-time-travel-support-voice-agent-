"""
FastAPI application: REST endpoints + the /ws/voice WebSocket protocol.

See docs/API.md for the full WebSocket event protocol.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.graph.workflow import build_graph
from app.observability import metrics as metrics_registry
from app.observability.logging import log_event
from app.observability.metrics import LatencyTimer, metrics_store
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.schemas import (
    ChatRequest, ChatResponse, HandoffInfo, HealthResponse, LatencyMetrics,
    RetrievedDocument, SynthesizeRequest, SynthesizeResponse, ToolCallRecord,
    TranscribeRequest, TranscribeResponse, TurnResult,
)
from app.services.audio import PlaybackController, base64_to_bytes, bytes_to_base64
from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService

settings = get_settings()

app = FastAPI(
    title="Real-Time Travel Support Voice Agent",
    description="Portfolio project: STT -> LangGraph agent (RAG + tools + handoff) -> TTS over WebSocket.",
    version="0.1.0",
)

# Shared singletons. In a larger deployment these would be built via
# FastAPI's dependency-injection system with per-request scoping where
# needed; for this project's size, module-level singletons are clearer.
vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
retriever = Retriever(vector_store)
llm_service = LLMService()
stt_service = STTService()
tts_service = TTSService()
graph = build_graph(retriever, llm_service, settings)

_playback_controllers: dict[str, PlaybackController] = {}


def _get_playback_controller(session_id: str) -> PlaybackController:
    return _playback_controllers.setdefault(session_id, PlaybackController())


def run_turn(session_id: str, turn_id: str, user_text: str, timer: LatencyTimer) -> TurnResult:
    # retrieval_ms, tool_ms, and llm_ms are measured *inside* the graph
    # nodes themselves (see app/graph/nodes.py) via this registry, so the
    # numbers reflect the actual stage doing the work rather than the
    # whole graph.invoke() call.
    metrics_registry.register_timer(turn_id, timer)
    try:
        initial_state = {"session_id": session_id, "turn_id": turn_id, "user_text": user_text}
        final_state = graph.invoke(initial_state)
    finally:
        metrics_registry.release_timer(turn_id)

    latency = LatencyMetrics(**timer.as_dict())
    metrics_store.record_turn(timer.as_dict())

    turn = TurnResult(
        session_id=session_id,
        turn_id=turn_id,
        user_text=user_text,
        response_text=final_state.get("response", ""),
        retrieved_documents=[RetrievedDocument(**d) for d in final_state.get("retrieved_documents", [])],
        tool_calls=[
            ToolCallRecord(tool_name=t["tool_name"], arguments=t["arguments"], result=t["result"],
                            success=t["result"].get("found", True) is not False, latency_ms=0.0)
            for t in final_state.get("tool_calls", [])
        ],
        confidence=final_state.get("confidence", 0.0),
        handoff=HandoffInfo(
            handoff_required=final_state.get("handoff_required", False),
            reason=final_state.get("handoff_reason"),
            priority=final_state.get("handoff_priority"),
            conversation_summary=user_text if final_state.get("handoff_required") else None,
        ),
        latency=latency,
    )
    return turn


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_mock_mode=settings.llm_mock_mode,
        tts_mock_mode=settings.tts_mock_mode,
        vector_store_ready=vector_store.is_ready,
    )


@app.get("/metrics")
def metrics() -> dict:
    return metrics_store.summary()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    timer = LatencyTimer()
    turn_id = str(uuid.uuid4())
    turn = run_turn(request.session_id, turn_id, request.text, timer)
    return ChatResponse(turn=turn)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    start = time.perf_counter()
    try:
        audio_bytes = base64_to_bytes(request.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 audio payload: {exc}") from exc
    text = stt_service.transcribe(audio_bytes)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return TranscribeResponse(text=text, is_mock=stt_service.is_mock_mode, latency_ms=latency_ms)


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    start = time.perf_counter()
    audio_bytes = tts_service.synthesize(request.text)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return SynthesizeResponse(
        audio_base64=bytes_to_base64(audio_bytes), is_mock=tts_service.is_mock_mode, latency_ms=latency_ms
    )


# --------------------------------------------------------------------------
# WebSocket voice protocol -- see docs/API.md
# --------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    playback = _get_playback_controller(session_id)

    await websocket.send_json({"type": "ready", "session_id": session_id, "payload": {}})
    log_event("session_started", session_id)

    try:
        while True:
            message = await websocket.receive_json()
            event_type = message.get("type")

            if event_type == "end":
                await websocket.send_json({"type": "end", "session_id": session_id, "payload": {}})
                break

            if event_type not in {"audio", "text"}:
                await websocket.send_json({
                    "type": "error", "session_id": session_id,
                    "payload": {"message": f"unsupported event type: {event_type}"},
                })
                continue

            # Barge-in: a new turn arriving while the previous one's audio
            # was still "playing" counts as an interruption.
            barge_in = playback.interrupt()

            turn_id = str(uuid.uuid4())
            timer = LatencyTimer()

            if event_type == "audio":
                with timer.measure("stt_ms"):
                    audio_bytes = base64_to_bytes(message["audio_base64"])
                    user_text = stt_service.transcribe(audio_bytes)
                await websocket.send_json({
                    "type": "transcript", "session_id": session_id, "turn_id": turn_id,
                    "payload": {"text": user_text, "is_mock_stt": stt_service.is_mock_mode},
                })
            else:
                user_text = message["text"]

            turn = run_turn(session_id, turn_id, user_text, timer)

            await websocket.send_json({
                "type": "agent_response", "session_id": session_id, "turn_id": turn_id,
                "payload": {
                    "text": turn.response_text,
                    "confidence": turn.confidence,
                    "used_doc_ids": [d.doc_id for d in turn.retrieved_documents],
                    "barge_in_detected": barge_in,
                },
            })

            with timer.measure("tts_ms"):
                audio_bytes = tts_service.synthesize(turn.response_text)
            playback.start(turn_id)
            await websocket.send_json({
                "type": "audio_response", "session_id": session_id, "turn_id": turn_id,
                "payload": {"audio_base64": bytes_to_base64(audio_bytes), "is_mock_tts": tts_service.is_mock_mode},
            })
            playback.finish(turn_id)

            await websocket.send_json({
                "type": "metrics", "session_id": session_id, "turn_id": turn_id,
                "payload": timer.as_dict(),
            })

            if turn.handoff.handoff_required:
                await websocket.send_json({
                    "type": "handoff", "session_id": session_id, "turn_id": turn_id,
                    "payload": turn.handoff.model_dump(),
                })

    except WebSocketDisconnect:
        log_event("session_disconnected", session_id)
    finally:
        _playback_controllers.pop(session_id, None)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=500, content={"error": str(exc)})
