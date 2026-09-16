"""Mocked rebooking-options tool, backed by synthetic in-memory availability."""
from __future__ import annotations

_SYNTHETIC_AVAILABILITY = {
    "AI-202": ["AI-204 (2026-10-03, 09:10)", "AI-206 (2026-10-03, 18:40)"],
    "AI-311": ["AI-313 (2026-09-21, 07:30)", "AI-315 (2026-09-21, 14:00)"],
    "AI-455": ["AI-457 (2026-09-15, 21:00)"],
}


def check_rebooking_options(flight_number: str) -> dict:
    options = _SYNTHETIC_AVAILABILITY.get(flight_number.upper(), [])
    return {"flight_number": flight_number.upper(), "options": options, "available": bool(options)}
