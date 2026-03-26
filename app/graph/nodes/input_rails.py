from __future__ import annotations

from app.graph.state import RAGState
from app.guardrails.input_guard import InputGuard

_guard = InputGuard()


def input_rails_node(state: RAGState) -> dict:
    """
    Step 2 — Gateway input guardrails.

    Checks the raw user query for:
    - Toxic/harmful content  → error_code "content_filtered"
    - Prompt injection       → error_code "injection_detected"

    On failure sets input_guard_passed=False and records the reason.
    """
    passed, reason = _guard.check(state["query"])

    if not passed:
        return {
            "input_guard_passed": False,
            "input_guard_reason": reason,
            "error": f"Input blocked: {reason}",
            "error_code": reason,
        }

    return {"input_guard_passed": True, "input_guard_reason": None}
