from app.tools.registry import execute_tool


def test_lookup_booking_status_found():
    result = execute_tool("lookup_booking_status", {"booking_id": "pnr1001"})
    assert result["found"] is True
    assert result["status"] == "confirmed"


def test_lookup_booking_status_not_found():
    result = execute_tool("lookup_booking_status", {"booking_id": "PNR9999"})
    assert result["found"] is False


def test_lookup_baggage_policy():
    result = execute_tool("lookup_baggage_policy", {})
    assert result["checked_kg"] == 23


def test_unknown_tool_returns_error():
    result = execute_tool("not_a_real_tool", {})
    assert "error" in result


def test_llm_service_selects_baggage_tool(llm_service):
    tool_calls = llm_service.select_and_execute_tool("What is my checked baggage allowance?")
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "lookup_baggage_policy"


def test_llm_service_extracts_booking_id(llm_service):
    tool_calls = llm_service.select_and_execute_tool("What's the status of booking PNR1002?")
    assert tool_calls[0]["tool_name"] == "lookup_booking_status"
    assert tool_calls[0]["arguments"]["booking_id"] == "PNR1002"
    assert tool_calls[0]["result"]["status"] == "cancelled_by_airline"
