def test_confidence_high_for_grounded_answer(graph):
    state = graph.invoke({
        "session_id": "s1", "turn_id": "t1",
        "user_text": "What is the checked baggage weight limit?",
    })
    assert state["confidence"] > 0.45


def test_confidence_low_for_vague_query(graph):
    state = graph.invoke({
        "session_id": "s2", "turn_id": "t2",
        "user_text": "What about my flight?",
    })
    assert state["confidence"] < 0.45


def test_confidence_is_between_zero_and_one(graph):
    for text in ["baggage allowance", "fraud on my card", "asdkjalskdj nonsense query"]:
        state = graph.invoke({"session_id": "s", "turn_id": text, "user_text": text})
        assert 0.0 <= state["confidence"] <= 1.0
