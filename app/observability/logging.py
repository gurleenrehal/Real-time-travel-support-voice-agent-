"""
Structured (JSON-lines) logging for the voice agent.

Every turn logs one structured record containing session_id, turn_id,
timestamp, event_type, latency, tool name, retrieval scores, and handoff
status -- never raw user audio or full transcript content, to avoid
logging sensitive user data.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_LOGGER_NAME = "voice_agent"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(
    event_type: str,
    session_id: str,
    turn_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit one structured JSON log line. Never pass raw transcript/audio."""
    record = {
        "timestamp": time.time(),
        "event_type": event_type,
        "session_id": session_id,
        "turn_id": turn_id,
        **fields,
    }
    get_logger().info(json.dumps(record, default=str))
