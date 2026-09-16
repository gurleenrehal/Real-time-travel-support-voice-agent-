"""
Mocked booking-status tool.

Backed by an in-memory synthetic "database" of bookings, not a real
airline reservation system. Designed so the lookup function's signature
(input: booking_id -> structured status dict) is exactly what a REST-API
-backed implementation would look like, so swapping in a real backend
later is a body-only change.
"""
from __future__ import annotations

_SYNTHETIC_BOOKINGS = {
    "PNR1001": {"status": "confirmed", "flight": "AI-202", "date": "2026-10-02", "seat": "14C"},
    "PNR1002": {"status": "cancelled_by_airline", "flight": "AI-311", "date": "2026-09-20", "seat": None},
    "PNR1003": {"status": "delayed", "flight": "AI-455", "date": "2026-09-15", "seat": "22A", "delay_minutes": 210},
}


def lookup_booking_status(booking_id: str) -> dict:
    booking = _SYNTHETIC_BOOKINGS.get(booking_id.upper())
    if booking is None:
        return {"found": False, "booking_id": booking_id}
    return {"found": True, "booking_id": booking_id.upper(), **booking}
