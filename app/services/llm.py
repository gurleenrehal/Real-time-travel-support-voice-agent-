"""
LLM service: two backends behind one interface.

- Mock mode (default, no OPENAI_API_KEY): a deterministic, extractive
  responder. It never invents policy text -- it either (a) grounds its
  answer in the single highest-scoring retrieved document plus any tool
  result, or (b) admits it doesn't have grounded information and
  recommends human handoff. This is what makes the system fully testable
  offline and keeps the "no fabricated policy" guarantee auditable.
- Real mode (OPENAI_API_KEY set): calls an OpenAI-compatible chat
  completion endpoint with function-calling enabled over TOOL_SCHEMAS,
  instructed via system prompt to answer only from the provided context
  and to say so explicitly when the context is insufficient.

Both modes return the same shape: {response_text, tool_calls, used_doc_ids,
unsupported}.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from app.config import get_settings
from app.tools.registry import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = (
    "You are a travel support voice assistant. Answer ONLY using the provided "
    "knowledge-base context and tool results. If the context does not contain "
    "the answer, say so plainly and do not invent airline policy. Keep answers "
    "to 2-3 sentences suitable for being spoken aloud."
)

_BOOKING_ID_RE = re.compile(r"\b(PNR\d{3,})\b", re.IGNORECASE)
_FLIGHT_NO_RE = re.compile(r"\b([A-Z]{2}-\d{2,4})\b", re.IGNORECASE)


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    def generate(self, user_text: str, retrieved_docs: list[dict]) -> dict[str, Any]:
        if self.settings.llm_mock_mode:
            return self._generate_mock(user_text, retrieved_docs)
        return self._generate_openai(user_text, retrieved_docs)

    # ------------------------------------------------------------------
    def _select_tool(self, user_text: str) -> dict | None:
        text_lower = user_text.lower()
        for schema in TOOL_SCHEMAS:
            if any(keyword in text_lower for keyword in schema["keywords"]):
                return schema
        return None

    def _extract_arguments(self, schema: dict, user_text: str) -> dict[str, str]:
        args: dict[str, str] = {}
        params = schema["parameters"]
        if "booking_id" in params:
            match = _BOOKING_ID_RE.search(user_text)
            args["booking_id"] = match.group(1).upper() if match else "UNKNOWN"
        if "flight_number" in params:
            match = _FLIGHT_NO_RE.search(user_text)
            args["flight_number"] = match.group(1).upper() if match else "UNKNOWN"
        if "description" in params:
            args["description"] = user_text
        if "reason" in params:
            args["reason"] = "explicit customer request" if "human" in user_text.lower() else "low confidence"
        if "priority" in params:
            args["priority"] = "medium"
        if "conversation_summary" in params:
            args["conversation_summary"] = user_text
        return args

    def select_and_execute_tool(self, user_text: str) -> list[dict[str, Any]]:
        """Step 1 of the mock pipeline, used directly by the LangGraph
        `tool_call_if_required` node so tool execution is its own,
        independently-loggable step."""
        tool_schema = self._select_tool(user_text)
        if tool_schema is None:
            return []
        args = self._extract_arguments(tool_schema, user_text)
        result = execute_tool(tool_schema["name"], args)
        return [{"tool_name": tool_schema["name"], "arguments": args, "result": result}]

    def compose_mock_response(
        self, user_text: str, retrieved_docs: list[dict], tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Step 2 of the mock pipeline, used by the LangGraph
        `generate_response` node. Never invents policy: it either grounds
        in the top retrieved document plus any tool result, or admits it
        has no grounded answer."""
        tool_result_text = ""
        if tool_calls:
            call = tool_calls[0]
            tool_result_text = f" Tool lookup ({call['tool_name']}) returned: {call['result']}."

        used_doc_ids = [d["doc_id"] for d in retrieved_docs]

        if retrieved_docs:
            best = retrieved_docs[0]
            response_text = f"Based on our travel-support policy: {best['text']}{tool_result_text}".strip()
            unsupported = False
        elif tool_calls:
            response_text = f"Here's what I found.{tool_result_text}".strip()
            unsupported = False
        else:
            response_text = (
                "I don't have grounded information on that in our current knowledge base, "
                "so I don't want to guess. I can connect you with a human support agent for this."
            )
            unsupported = True

        return {
            "response_text": response_text,
            "tool_calls": tool_calls,
            "used_doc_ids": used_doc_ids,
            "unsupported": unsupported,
        }

    def _generate_mock(self, user_text: str, retrieved_docs: list[dict]) -> dict[str, Any]:
        tool_calls = self.select_and_execute_tool(user_text)
        return self.compose_mock_response(user_text, retrieved_docs, tool_calls)

    # ------------------------------------------------------------------
    def _generate_openai(self, user_text: str, retrieved_docs: list[dict]) -> dict[str, Any]:
        """Real-LLM path. Requires network access to api.openai.com and a
        valid OPENAI_API_KEY; not exercised by the test suite or in this
        sandboxed environment (see README limitations)."""
        context = "\n".join(f"- {d['text']}" for d in retrieved_docs) or "(no relevant context retrieved)"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_text}"},
        ]
        functions = [
            {"name": s["name"], "description": s["description"],
             "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in s["parameters"]}}}
            for s in TOOL_SCHEMAS
        ]
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": self.settings.openai_model,
                    "messages": messages,
                    "functions": functions,
                    "temperature": 0.2,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            tool_calls: list[dict[str, Any]] = []
            function_call = choice.get("function_call")
            if function_call:
                import json as _json
                name = function_call["name"]
                args = _json.loads(function_call.get("arguments") or "{}")
                result = execute_tool(name, args)
                tool_calls.append({"tool_name": name, "arguments": args, "result": result})
            response_text = choice.get("content") or "(tool call executed; awaiting follow-up turn)"
            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "used_doc_ids": [d["doc_id"] for d in retrieved_docs],
                "unsupported": not retrieved_docs and not tool_calls,
            }
        except Exception as exc:  # network unavailable, bad key, etc.
            return {
                "response_text": (
                    "I'm having trouble reaching the language model right now. "
                    "I can connect you with a human support agent instead."
                ),
                "tool_calls": [],
                "used_doc_ids": [],
                "unsupported": True,
                "error": str(exc),
            }
