"""
Tool registry: maps tool name -> (callable, JSON-schema description).

This is what both the mock LLM and a real OpenAI-function-calling LLM
consult to decide which tool to invoke and with what arguments.
"""
from __future__ import annotations

from typing import Any, Callable

from app.tools.baggage import lookup_baggage_policy, report_lost_baggage
from app.tools.booking import lookup_booking_status
from app.tools.handoff import create_handoff_request
from app.tools.rebooking import check_rebooking_options

ToolFunc = Callable[..., dict[str, Any]]

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_booking_status",
        "description": "Look up the status of a booking by its PNR/booking id.",
        "keywords": ["booking", "pnr", "status", "reservation", "flight status"],
        "parameters": {"booking_id": "string"},
    },
    {
        "name": "lookup_baggage_policy",
        "description": "Return the standard checked/cabin baggage allowance and excess fees.",
        "keywords": ["baggage allowance", "checked bag", "cabin bag", "excess baggage"],
        "parameters": {},
    },
    {
        "name": "report_lost_baggage",
        "description": "File a Property Irregularity Report for lost or delayed baggage.",
        "keywords": ["lost baggage", "missing bag", "baggage did not arrive", "pir"],
        "parameters": {"booking_id": "string", "description": "string"},
    },
    {
        "name": "check_rebooking_options",
        "description": "Check available rebooking flight options for a given flight number.",
        "keywords": ["rebook", "rebooking", "alternative flight", "next available flight"],
        "parameters": {"flight_number": "string"},
    },
    {
        "name": "create_handoff_request",
        "description": "Create a human-agent handoff ticket.",
        "keywords": ["human agent", "speak to a person", "escalate"],
        "parameters": {"reason": "string", "priority": "string", "conversation_summary": "string"},
    },
]

TOOL_REGISTRY: dict[str, ToolFunc] = {
    "lookup_booking_status": lookup_booking_status,
    "lookup_baggage_policy": lookup_baggage_policy,
    "report_lost_baggage": report_lost_baggage,
    "check_rebooking_options": check_rebooking_options,
    "create_handoff_request": create_handoff_request,
}


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return {"error": f"unknown tool: {tool_name}"}
    return func(**arguments)
