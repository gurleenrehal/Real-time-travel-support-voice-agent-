def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_mock_mode"] is True
    assert body["tts_mock_mode"] is True
    assert body["vector_store_ready"] is True


def test_metrics_endpoint_empty_is_ok(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "turn_count" in resp.json()
