#!/usr/bin/env python3
"""
Evaluation harness: runs data/evaluation/eval_set.json through the actual
LangGraph pipeline (same graph the FastAPI app uses) and computes:

  1. Retrieval precision / hit rate      (vs. hand-labeled expected_relevant_doc_ids)
  2. Retrieval relevance                 (avg similarity score of correctly-retrieved docs)
  3. Tool-call success rate              (correct tool selected when one was expected)
  4. Handoff accuracy                    (predicted handoff vs. expected_handoff)
  5. Response groundedness               (answerable cases that got a grounded, non-"unsupported" answer)
  6. Unsupported-claim rate              (how often the agent said "I don't know" -- a proxy for
                                           hallucination risk, since the mock LLM is extractive and
                                           architecturally cannot invent policy text; a real-LLM
                                           deployment would need a separate factuality check against
                                           this same signal)
  7. STT accuracy                        NOT EVALUATED -- no real audio/reference-transcript pairs
                                           exist in this sandboxed project (see README limitations).
                                           Reported as null rather than fabricated.
  8. End-to-end latency                  (per-stage + total, from the real LatencyTimer)

Nothing here is fabricated: every number is computed from an actual
graph.invoke() call per test case. Run with: python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.graph.workflow import build_graph
from app.observability import metrics as metrics_registry
from app.observability.metrics import LatencyTimer
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.llm import LLMService


def load_eval_set(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["cases"]


def run_case(graph, case: dict) -> dict:
    turn_id = str(uuid.uuid4())
    timer = LatencyTimer()
    metrics_registry.register_timer(turn_id, timer)
    try:
        state = graph.invoke({
            "session_id": "eval-session", "turn_id": turn_id, "user_text": case["user_text"],
        })
    finally:
        metrics_registry.release_timer(turn_id)

    retrieved_ids = [d["doc_id"] for d in state.get("retrieved_documents", [])]
    retrieved_scores = state.get("retrieval_scores", [])
    tool_names = [t["tool_name"] for t in state.get("tool_calls", [])]
    unsupported = state.get("unsupported", False)
    handoff_required = state.get("handoff_required", False)

    expected_ids = set(case["expected_relevant_doc_ids"])
    retrieved_id_set = set(retrieved_ids)

    hit = bool(expected_ids & retrieved_id_set) if expected_ids else (len(retrieved_id_set) == 0)
    precision = (
        len(expected_ids & retrieved_id_set) / len(retrieved_id_set)
        if retrieved_id_set else (1.0 if not expected_ids else 0.0)
    )

    expected_tool = case.get("expected_tool")
    tool_correct = (expected_tool in tool_names) if expected_tool else (len(tool_names) == 0)

    handoff_correct = handoff_required == case["expect_handoff"]

    is_answerable = case["is_answerable"]
    grounded = is_answerable and not unsupported

    return {
        "id": case["id"],
        "category": case["category"],
        "retrieval_hit": hit,
        "retrieval_precision": precision,
        "retrieval_avg_score": (sum(retrieved_scores) / len(retrieved_scores)) if retrieved_scores else None,
        "tool_correct": tool_correct,
        "handoff_correct": handoff_correct,
        "handoff_predicted": handoff_required,
        "is_answerable": is_answerable,
        "unsupported": unsupported,
        "grounded_when_answerable": grounded if is_answerable else None,
        "confidence": state.get("confidence"),
        "latency_ms": timer.as_dict(),
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)

    def pct(pred) -> float:
        vals = [pred(r) for r in results]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    answerable = [r for r in results if r["is_answerable"]]
    unanswerable = [r for r in results if not r["is_answerable"]]

    latency_keys = {k for r in results for k in r["latency_ms"]}
    latency_summary = {}
    for key in latency_keys:
        vals = [r["latency_ms"][key] for r in results if key in r["latency_ms"]]
        if vals:
            latency_summary[f"{key}_avg"] = round(sum(vals) / len(vals), 3)
            latency_summary[f"{key}_max"] = round(max(vals), 3)

    return {
        "total_cases": n,
        "retrieval_hit_rate": pct(lambda r: 1.0 if r["retrieval_hit"] else 0.0),
        "retrieval_precision_avg": pct(lambda r: r["retrieval_precision"]),
        "retrieval_relevance_avg_score": pct(lambda r: r["retrieval_avg_score"]),
        "tool_call_success_rate": pct(lambda r: 1.0 if r["tool_correct"] else 0.0),
        "handoff_accuracy": pct(lambda r: 1.0 if r["handoff_correct"] else 0.0),
        "response_groundedness_rate_on_answerable": (
            round(sum(1 for r in answerable if r["grounded_when_answerable"]) / len(answerable), 4)
            if answerable else None
        ),
        "unsupported_claim_rate_overall": pct(lambda r: 1.0 if r["unsupported"] else 0.0),
        "correct_unsupported_rate_on_unanswerable": (
            round(sum(1 for r in unanswerable if r["unsupported"]) / len(unanswerable), 4)
            if unanswerable else None
        ),
        "stt_accuracy": None,  # not evaluated -- see module docstring
        "stt_accuracy_note": "Not evaluated: no real audio + reference-transcript pairs available in this environment.",
        "latency_ms_summary": latency_summary,
    }


def main() -> None:
    settings = get_settings()
    vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
    if not vector_store.is_ready:
        print("Vector store is empty -- run `python scripts/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    retriever = Retriever(vector_store)
    llm_service = LLMService()
    graph = build_graph(retriever, llm_service, settings)

    cases = load_eval_set(settings.evaluation_set_path)
    results = [run_case(graph, case) for case in cases]
    summary = summarize(results)

    report = {"summary": summary, "cases": results}
    Path(settings.evaluation_report_path).write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(summary, indent=2))
    print(f"\nFull report written to {settings.evaluation_report_path}")


if __name__ == "__main__":
    main()
