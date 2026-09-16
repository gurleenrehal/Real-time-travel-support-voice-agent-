"""Creates a structured human-handoff ticket (synthetic; no real ticketing
system is integrated -- this returns a structured record that a real
support-desk integration would consume)."""
from __future__ import annotations

_TICKETS: list[dict] = []


def create_handoff_request(reason: str, priority: str, conversation_summary: str) -> dict:
    ticket_id = f"HANDOFF-{len(_TICKETS) + 1:04d}"
    ticket = {
        "ticket_id": ticket_id,
        "reason": reason,
        "priority": priority,
        "conversation_summary": conversation_summary,
        "status": "queued",
    }
    _TICKETS.append(ticket)
    return ticket
