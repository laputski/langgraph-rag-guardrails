from __future__ import annotations

from app.graph.state import RAGState
from app.guardrails.output_guard import OutputGuard


def make_output_guardrails_node(output_guard: OutputGuard, max_retries: int = 2):
    """
    Step 10 — Output guardrails.

    Validates the LLM answer before it is delivered to the user:
    1. Toxicity check — blocks harmful generated content
    2. Hallucination check — flags answers not grounded in retrieved context

    On failure: if retry_count < max_retries, the graph loops back to
    llm_generate for another attempt. Otherwise, sets max_retries_exceeded error.
    """
    def output_guardrails_node(state: RAGState) -> dict:
        answer = state.get("llm_answer") or ""
        context = state.get("context_text") or ""

        passed, reason = output_guard.check(answer, context)

        if passed:
            # Extract unique source titles for the response
            sources = list(
                dict.fromkeys(c.title for c in state.get("reranked_chunks", []))
            )
            return {
                "output_guard_passed": True,
                "output_guard_reason": None,
                "final_answer": answer,
                "sources": sources,
            }

        # Failed — check if we've exhausted retries
        retry_count = state.get("llm_retry_count", 0)
        if retry_count >= max_retries:
            return {
                "output_guard_passed": False,
                "output_guard_reason": reason,
                "error": f"Output guardrail failed after {max_retries} retries: {reason}",
                "error_code": "max_retries_exceeded",
            }

        # Still have retries available — will loop back to llm_generate
        return {
            "output_guard_passed": False,
            "output_guard_reason": reason,
        }

    return output_guardrails_node
