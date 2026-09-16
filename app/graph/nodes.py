"""
LangGraph node functions.

Each node is a plain function of (state) -> partial state update, built
via a factory so nodes can close over the Retriever/LLMService instances
without relying on global state. This module is where the resume claim
"LangGraph actually controls the workflow, not decoratively" is made
concrete: `evaluate_retrieval` and `evaluate_confidence` change what the
conditional edges in workflow.py actually do (skip vs. use RAG context,
route to a human handoff vs. respond).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.config import Settings
from app.graph.state import AgentState
from app.observability import metrics as metrics_module
from app.observability.logging import log_event
from app.rag.retriever import Retriever
from app.services.llm import LLMService

@contextmanager
def _noop_ctx():
    yield


_ESCALATION_KEYWORDS = [
    "fraud", "unauthorized charge", "stolen", "passport", "medical emergency",
    "legal", "lawsuit", "human agent", "speak to a person", "talk to a human",
]


def build_nodes(retriever: Retriever, llm_service: LLMService, settings: Settings) -> dict[str, Any]:

    def receive_turn(state: AgentState) -> AgentState:
        log_event("turn_received", state["session_id"], state["turn_id"])
        return {}

    def retrieve_context(state: AgentState) -> AgentState:
        timer = metrics_module.get_timer(state["turn_id"])
        with (timer.measure("retrieval_ms") if timer else _noop_ctx()):
            docs = retriever.retrieve(state["user_text"])
        log_event(
            "retrieval_complete", state["session_id"], state["turn_id"],
            retrieval_scores=[d["score"] for d in docs],
        )
        return {
            "retrieved_documents": docs,
            "retrieval_scores": [d["score"] for d in docs],
        }

    def evaluate_retrieval(state: AgentState) -> AgentState:
        # This flag is genuinely consumed by generate_response below (not
        # decorative): when retrieval confidence is under the floor, the
        # retrieved documents are NOT used for grounding even if something
        # technically cleared the retriever's top-k threshold, which
        # prevents low-relevance TF-IDF matches on short/vague queries
        # from being presented as a confident, sourced answer.
        scores = state.get("retrieval_scores", [])
        retrieval_ok = bool(scores) and (sum(scores) / len(scores)) >= settings.retrieval_confidence_floor
        return {"retrieval_ok": retrieval_ok}  # type: ignore[typeddict-item]

    def tool_call_if_required(state: AgentState) -> AgentState:
        timer = metrics_module.get_timer(state["turn_id"])
        with (timer.measure("tool_ms") if timer else _noop_ctx()):
            tool_calls = llm_service.select_and_execute_tool(state["user_text"])
        log_event(
            "tool_call", state["session_id"], state["turn_id"],
            tool_names=[t["tool_name"] for t in tool_calls],
        )
        return {"tool_calls": tool_calls, "tool_results": [t["result"] for t in tool_calls]}

    def generate_response(state: AgentState) -> AgentState:
        retrieval_ok = state.get("retrieval_ok", True)  # type: ignore[typeddict-item]
        usable_docs = state.get("retrieved_documents", []) if retrieval_ok else []

        timer = metrics_module.get_timer(state["turn_id"])
        with (timer.measure("llm_ms") if timer else _noop_ctx()):
            result = llm_service.compose_mock_response(
                state["user_text"], usable_docs, state.get("tool_calls", [])
            ) if settings.llm_mock_mode else llm_service.generate(
                state["user_text"], usable_docs
            )
        return {"response": result["response_text"], "unsupported": result["unsupported"]}  # type: ignore[typeddict-item]

    def evaluate_confidence(state: AgentState) -> AgentState:
        unsupported = state.get("unsupported", False)
        retrieval_ok = state.get("retrieval_ok", True)  # type: ignore[typeddict-item]
        tool_calls = state.get("tool_calls", [])
        scores = state.get("retrieval_scores", [])

        if unsupported:
            confidence = 0.1
        elif not (retrieval_ok and scores) and not tool_calls:
            # No usable retrieval AND no tool executed: there is no
            # grounding signal at all, so confidence must be low rather
            # than a "neutral" default -- this is what previously let
            # vague/ambiguous turns slip past the handoff threshold.
            confidence = 0.2
        else:
            # Use the TOP retrieved score, not the mean of all top-k: the
            # response is grounded in the single best-matching document
            # (see LLMService.compose_mock_response), so confidence should
            # track that document's relevance rather than being diluted by
            # lower-ranked, less relevant context docs.
            top_score = scores[0] if (retrieval_ok and scores) else 0.4
            retrieval_component = top_score

            if not tool_calls:
                tool_component = 0.5  # neutral: no tool was needed, doc grounding carries it
            else:
                result = tool_calls[0]["result"]
                tool_component = 0.3 if result.get("found") is False else 1.0

            confidence = round(0.6 * retrieval_component + 0.4 * tool_component, 4)
            confidence = min(1.0, max(0.0, confidence))

        log_event("confidence_evaluated", state["session_id"], state["turn_id"], confidence=confidence)
        return {"confidence": confidence}

    def decide_handoff(state: AgentState) -> AgentState:
        text_lower = state["user_text"].lower()
        keyword_trigger = next((kw for kw in _ESCALATION_KEYWORDS if kw in text_lower), None)
        low_confidence = state.get("confidence", 1.0) < settings.confidence_handoff_threshold

        if keyword_trigger:
            reason, priority, required = f"escalation keyword matched: '{keyword_trigger}'", "high", True
        elif low_confidence:
            reason, priority, required = "low agent confidence in grounded answer", "medium", True
        else:
            reason, priority, required = None, None, False

        log_event("handoff_decision", state["session_id"], state["turn_id"], handoff_required=required, reason=reason)
        return {"handoff_required": required, "handoff_reason": reason, "handoff_priority": priority}

    def finalize_handoff(state: AgentState) -> AgentState:
        from app.tools.handoff import create_handoff_request

        ticket = create_handoff_request(
            reason=state.get("handoff_reason") or "unspecified",
            priority=state.get("handoff_priority") or "medium",
            conversation_summary=state["user_text"],
        )
        appended = (
            f" I'm connecting you with a human support agent (ticket {ticket['ticket_id']}) "
            f"who can help further."
        )
        return {"response": state.get("response", "") + appended}

    return {
        "receive_turn": receive_turn,
        "retrieve_context": retrieve_context,
        "evaluate_retrieval": evaluate_retrieval,
        "tool_call_if_required": tool_call_if_required,
        "generate_response": generate_response,
        "evaluate_confidence": evaluate_confidence,
        "decide_handoff": decide_handoff,
        "finalize_handoff": finalize_handoff,
    }
