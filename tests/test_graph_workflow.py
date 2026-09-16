def test_graph_produces_all_expected_state_keys(graph):
    state = graph.invoke({
        "session_id": "s1", "turn_id": "t1",
        "user_text": "What is the checked baggage weight limit?",
    })
    for key in ["response", "confidence", "handoff_required", "retrieved_documents", "retrieval_scores"]:
        assert key in state


def test_graph_routes_to_finalize_handoff_when_required(graph):
    state = graph.invoke({
        "session_id": "s2", "turn_id": "t2",
        "user_text": "I lost my passport, this is urgent.",
    })
    assert state["handoff_required"] is True
    # finalize_handoff appends a ticket reference -- proves the
    # conditional edge actually ran, not just decided.
    assert "HANDOFF-" in state["response"]


def test_graph_skips_finalize_handoff_when_not_required(graph):
    state = graph.invoke({
        "session_id": "s3", "turn_id": "t3",
        "user_text": "What is the checked baggage weight limit?",
    })
    assert state["handoff_required"] is False
    assert "HANDOFF-" not in state["response"]


def test_tool_call_populates_tool_results(graph):
    state = graph.invoke({
        "session_id": "s4", "turn_id": "t4",
        "user_text": "What is my baggage allowance?",
    })
    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0]["tool_name"] == "lookup_baggage_policy"
