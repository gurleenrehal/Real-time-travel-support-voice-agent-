def test_handoff_triggered_by_escalation_keyword(graph):
    state = graph.invoke({
        "session_id": "s1", "turn_id": "t1",
        "user_text": "I think there was an unauthorized charge on my card, this looks like fraud.",
    })
    assert state["handoff_required"] is True
    assert "fraud" in state["handoff_reason"]


def test_handoff_triggered_by_explicit_request(graph):
    state = graph.invoke({
        "session_id": "s2", "turn_id": "t2",
        "user_text": "I want to speak to a human agent right now.",
    })
    assert state["handoff_required"] is True


def test_no_handoff_for_clear_grounded_question(graph):
    state = graph.invoke({
        "session_id": "s3", "turn_id": "t3",
        "user_text": "What is the checked baggage weight limit?",
    })
    assert state["handoff_required"] is False


def test_handoff_response_includes_ticket_reference(graph):
    state = graph.invoke({
        "session_id": "s4", "turn_id": "t4",
        "user_text": "I want to speak to a human agent.",
    })
    assert "ticket" in state["response"].lower()
    assert "HANDOFF-" in state["response"]
