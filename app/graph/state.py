"""Typed state threaded through the LangGraph workflow."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    turn_id: str
    user_text: str

    retrieved_documents: list[dict[str, Any]]
    retrieval_scores: list[float]
    retrieval_ok: bool

    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    response: str
    unsupported: bool
    confidence: float
    handoff_required: bool
    handoff_reason: str | None
    handoff_priority: str | None

    latency_metrics: dict[str, float]
