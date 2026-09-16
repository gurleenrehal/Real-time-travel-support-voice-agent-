"""Mocked baggage tools, backed by synthetic in-memory data."""
from __future__ import annotations

_BAGGAGE_ALLOWANCE = {"checked_kg": 23, "cabin_kg": 7, "excess_fee_per_kg": 500}

_SYNTHETIC_LOST_BAGGAGE_REPORTS: dict[str, dict] = {}


def lookup_baggage_policy() -> dict:
    return dict(_BAGGAGE_ALLOWANCE)


def report_lost_baggage(booking_id: str, description: str) -> dict:
    """Files a synthetic Property Irregularity Report (PIR)."""
    pir_id = f"PIR-{len(_SYNTHETIC_LOST_BAGGAGE_REPORTS) + 1:04d}"
    record = {"pir_id": pir_id, "booking_id": booking_id.upper(), "description": description, "status": "open"}
    _SYNTHETIC_LOST_BAGGAGE_REPORTS[pir_id] = record
    return record
