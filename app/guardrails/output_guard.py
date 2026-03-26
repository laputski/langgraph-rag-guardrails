from __future__ import annotations

import re
from typing import Optional

from app.guardrails.input_guard import _TOXIC_PATTERNS


# Phrases that indicate the model correctly admitted ignorance.
# These answers always pass guardrails regardless of token overlap.
_IGNORANCE_PHRASES = [
    "i don't have enough information",
    "i don't have information",
    "i cannot answer",
    "i can't answer",
    "i do not have",
    "not available in",
    "not covered in",
    "no information about",
]


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokenizer, strips punctuation."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {w for w in text.split() if len(w) > 2}


class OutputGuard:
    """
    Validates LLM-generated answers before delivering them to the user.

    Checks (in order):
    1. Toxicity         — blocks harmful generated content
    2. Hallucination    — flags answers not grounded in the retrieved context

    Hallucination detection uses a simple token-overlap heuristic:
        overlap = |answer_tokens ∩ context_tokens| / |answer_tokens|
    If overlap < threshold AND the answer is not an ignorance phrase,
    the response is considered potentially hallucinated and flagged for retry.

    In production, replace the heuristic with an NLI-based entailment model
    or an LLM-as-judge call (e.g., "Does this answer follow from this context?").
    """

    def __init__(self, hallucination_threshold: float = 0.25) -> None:
        self._threshold = hallucination_threshold

    def check(
        self,
        answer: str,
        context: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (passed, reason).

        passed=True means the answer is safe to deliver.
        If passed=False, reason is one of:
            "toxic_output"  → generated answer contains harmful content
            "hallucination" → answer tokens not sufficiently grounded in context
        """
        # Step 1: toxicity check on generated text
        if any(p.search(answer) for p in _TOXIC_PATTERNS):
            return False, "toxic_output"

        # Step 2: hallucination check
        if self._is_hallucinated(answer, context):
            return False, "hallucination"

        return True, None

    def _is_hallucinated(self, answer: str, context: str) -> bool:
        answer_lower = answer.lower().strip()

        # Short answers or ignorance admissions always pass
        if any(phrase in answer_lower for phrase in _IGNORANCE_PHRASES):
            return False

        answer_tokens = _tokenize(answer)
        if len(answer_tokens) <= 20:
            # Very short answers are hard to judge by overlap alone
            return False

        context_tokens = _tokenize(context)
        if not context_tokens:
            return False

        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return overlap < self._threshold
