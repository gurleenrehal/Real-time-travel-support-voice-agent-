def test_chat_missing_field_returns_422(client):
    resp = client.post("/chat", json={"session_id": "s1"})  # missing "text"
    assert resp.status_code == 422


def test_transcribe_invalid_base64_handled(client):
    resp = client.post("/transcribe", json={
        "session_id": "s1", "audio_base64": "not-valid-base64!!!", "sample_rate": 16000,
    })
    assert resp.status_code == 400


def test_synthesize_empty_text_still_returns_audio(client):
    resp = client.post("/synthesize", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json()["is_mock"] is True


def test_chat_endpoint_happy_path(client):
    resp = client.post("/chat", json={"session_id": "s1", "text": "What is the checked baggage weight limit?"})
    assert resp.status_code == 200
    turn = resp.json()["turn"]
    assert turn["session_id"] == "s1"
    assert 0.0 <= turn["confidence"] <= 1.0
