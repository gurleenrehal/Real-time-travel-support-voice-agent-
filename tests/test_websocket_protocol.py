import base64


def test_ws_ready_then_text_turn(client):
    with client.websocket_connect("/ws/voice") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        session_id = ready["session_id"]

        ws.send_json({"type": "text", "text": "What is the checked baggage weight limit?"})

        agent_response = ws.receive_json()
        assert agent_response["type"] == "agent_response"
        assert agent_response["session_id"] == session_id
        assert "23" in agent_response["payload"]["text"] or "baggage" in agent_response["payload"]["text"].lower()

        audio_response = ws.receive_json()
        assert audio_response["type"] == "audio_response"
        assert audio_response["payload"]["is_mock_tts"] is True
        # confirm it's valid base64-decodable audio bytes
        base64.b64decode(audio_response["payload"]["audio_base64"])

        metrics_event = ws.receive_json()
        assert metrics_event["type"] == "metrics"
        assert "total_ms" in metrics_event["payload"]

        ws.send_json({"type": "end"})
        end_event = ws.receive_json()
        assert end_event["type"] == "end"


def test_ws_audio_event_uses_mock_stt(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()  # ready
        fake_audio = base64.b64encode(b"what is the checked baggage weight limit").decode("ascii")
        ws.send_json({"type": "audio", "audio_base64": fake_audio})

        transcript_event = ws.receive_json()
        assert transcript_event["type"] == "transcript"
        assert transcript_event["payload"]["is_mock_stt"] is True
        assert "baggage" in transcript_event["payload"]["text"].lower()


def test_ws_handoff_event_sent_for_escalation(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "text", "text": "I want to speak to a human agent."})
        ws.receive_json()  # agent_response
        ws.receive_json()  # audio_response
        ws.receive_json()  # metrics
        handoff_event = ws.receive_json()
        assert handoff_event["type"] == "handoff"
        assert handoff_event["payload"]["handoff_required"] is True


def test_ws_unsupported_event_type_returns_error(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "not_a_real_event"})
        error_event = ws.receive_json()
        assert error_event["type"] == "error"
