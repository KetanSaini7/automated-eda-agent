"""
query_interpreter.py
====================
Converts natural-language questions into validated query plans using Gemini.

Design (from prompt.md §8):
  - Sends ONLY compact schema JSON to Gemini — never raw dataset rows.
  - Requests structured JSON output via system prompt.
  - Validates the returned plan before returning it.
  - Reuses _build_client() and _call_with_retry() from llm_agent.py.
  - Never duplicates Gemini client initialization.
  - Handles: invalid JSON, missing 'operation', unsupported questions,
    clarification requests, API errors, rate limits.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the system prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "query_interpreter_prompt.txt"

# Maximum characters of history to include in prompt (keep it compact)
_MAX_HISTORY_CHARS = 1500


# Module-level cache for the prompt template (loaded once, reused across questions).
# Reset to None whenever the prompt file changes so the new content is reloaded.
_PROMPT_TEMPLATE_CACHE: str | None = None


def _load_prompt_template() -> str:
    """Load the query interpreter system prompt template (cached after first load)."""
    global _PROMPT_TEMPLATE_CACHE
    if _PROMPT_TEMPLATE_CACHE is not None:
        return _PROMPT_TEMPLATE_CACHE
    try:
        _PROMPT_TEMPLATE_CACHE = _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("query_interpreter_prompt.txt not found; using minimal fallback.")
        _PROMPT_TEMPLATE_CACHE = (
            "Convert the user question to a JSON query plan. "
            "Schema: {SCHEMA_JSON}. History: {HISTORY}. "
            "Return ONLY valid JSON."
        )
    return _PROMPT_TEMPLATE_CACHE


def _compact_history(conversation_history: list[dict]) -> str:
    """Format recent conversation history compactly for the prompt."""
    if not conversation_history:
        return "(no prior conversation)"
    lines = []
    # Include only the last 6 turns to keep the prompt small
    for turn in conversation_history[-6:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))[:300]
        lines.append(f"{role.upper()}: {content}")
    text = "\n".join(lines)
    # Hard cap
    if len(text) > _MAX_HISTORY_CHARS:
        text = text[-_MAX_HISTORY_CHARS:]
    return text


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object from a string.
    Handles Gemini responses that wrap JSON in markdown code fences.
    """
    if not text:
        return ""
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Find the first { ... } block
    start = text.find("{")
    if start == -1:
        return ""
    # Walk to find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def interpret_question(
    question: str,
    schema: dict,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Convert a natural-language question into a structured query plan.

    Parameters
    ----------
    question : str
        The user's question about their dataset.
    schema : dict
        Compact schema from query_engine.build_schema().
    conversation_history : list[dict], optional
        Prior conversation turns [{role, content}, ...].

    Returns
    -------
    dict — one of:
      - Valid query plan:         {"operation": "...", ...}
      - Clarification needed:     {"clarification_needed": "..."}
      - Unsupported question:     {"unsupported": "..."}
      - Error:                    {"error": "..."}
    """
    # Deferred import to reuse existing initialized client infrastructure
    from src.llm_agent import (
        _build_client,
        _call_with_retry,
        _extract_text_from_interaction,
        _extract_text_from_response,
        MODEL_NAME,
    )

    if not question or not question.strip():
        return {"error": "Empty question provided."}

    if not schema:
        return {"error": "No dataset schema available. Please upload a dataset first."}

    history = conversation_history or []

    # Build the prompt
    template = _load_prompt_template()
    schema_json = json.dumps(schema, ensure_ascii=False, indent=None)
    # Keep schema compact — truncate if huge
    if len(schema_json) > 4000:
        # Send only column names and roles if schema is very large
        compact_cols = [
            {"name": c["name"], "role": c["role"], "dtype": c["dtype"]}
            for c in schema.get("columns", [])
        ]
        schema_json = json.dumps({
            "row_count": schema.get("row_count"),
            "column_count": schema.get("column_count"),
            "columns": compact_cols,
            "numeric_columns": schema.get("numeric_columns", []),
            "categorical_columns": schema.get("categorical_columns", []),
            "datetime_columns": schema.get("datetime_columns", []),
        }, ensure_ascii=False)

    prompt_text = template.replace("{SCHEMA_JSON}", schema_json).replace(
        "{HISTORY}", _compact_history(history)
    )

    # Combine system prompt + user question into a single user message
    # (Interactions API treats first message as the full context)
    full_message = f"{prompt_text}\n\n## USER QUESTION\n{question.strip()}"

    try:
        client = _build_client()
    except Exception as exc:
        logger.error("Failed to build Gemini client: %s", exc)
        return {"error": f"Gemini client initialization failed: {exc}"}

    import time as _time
    _t0 = _time.perf_counter()
    try:
        # Try Interactions API first (native gemini-3.8-flash path with thinking)
        try:
            interaction = _call_with_retry(
                client.interactions.create,
                model=MODEL_NAME,
                input=full_message,
                generation_config={"thinking_level": "low"},
            )
            raw_response = _extract_text_from_interaction(interaction)
        except Exception:
            raw_response = None

        # Fallback to generate_content if interactions returned nothing
        if not raw_response:
            response = _call_with_retry(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=full_message,
            )
            raw_response = _extract_text_from_response(response)
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        return {"error": f"Gemini API error: {exc}"}
    finally:
        logger.debug(
            "interpret_question Gemini call took %.2f s",
            _time.perf_counter() - _t0,
        )

    if not raw_response or not raw_response.strip():
        return {"error": "Gemini returned an empty response."}

    # Extract and parse JSON
    json_str = _extract_json(raw_response)
    if not json_str:
        logger.warning("No JSON found in Gemini response: %r", raw_response[:200])
        return {"error": "Could not parse a query plan from Gemini's response."}

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s | raw: %r", exc, json_str[:200])
        return {"error": f"Gemini returned invalid JSON: {exc}"}

    if not isinstance(plan, dict):
        return {"error": "Gemini returned a non-object JSON value."}

    # Pass through special response types without further validation
    if "clarification_needed" in plan or "unsupported" in plan or "error" in plan:
        return plan

    # Must have 'operation' field for a real query plan
    if "operation" not in plan:
        return {"error": "Gemini response is missing the 'operation' field."}

    return plan
