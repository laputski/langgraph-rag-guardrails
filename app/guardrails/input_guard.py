from __future__ import annotations

import re
from typing import Optional


# ── Toxicity patterns ─────────────────────────────────────────────────────────
# Keyword/regex list for harmful input detection.
# In production, replace with a dedicated classifier (e.g., Detoxify, Perspective API).
_TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bkill\s+(yourself|myself|all|them|him|her|everyone)\b",
        r"\b(bomb|explosive|weapon)\s+(instructions|how\s+to|make|build)\b",
        r"\b(n[i1]gg[ae]r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg[o0]t)\b",
        r"\b(hate|exterminate|genocide)\s+(all\s+)?(jews?|muslims?|christians?|blacks?|whites?)\b",
        r"\b(rape|sexual\s+assault)\s+(instructions?|how\s+to)\b",
        r"\bchild\s+(porn|pornography|sexual\s+abuse\s+material|sex)\b",
        r"\b(shoot|stab|murder)\s+(everyone|all\s+of\s+them|my\s+(boss|coworkers?))\b",
    ]
]

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        r"forget\s+(your|all|the)\s+(instructions?|rules?|context|training|guidelines?|previous)",
        r"you\s+are\s+now\s+(a|an|the|DAN|evil|unrestricted|jailbroken)",
        r"\bDAN\b.*\bmode\b",
        r"act\s+as\s+(if\s+you\s+(are|were)|a\s+different|an?\s+unrestricted|an?\s+evil)",
        r"(pretend|imagine|roleplay|simulate)\s+(you\s+are|being|that\s+you('re|\s+are))\s+(not\s+an?\s+AI|a\s+human|unrestricted)",
        r"system\s*:\s*you\s+are",
        r"\n\s*(human|user|assistant|system)\s*:",
        r"jailbreak",
        r"override\s+(your\s+)?(safety|ethical|content)\s+(filter|guidelines?|restrictions?|rules?)",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions?|training|hidden)",
        r"print\s+(your\s+)?(system\s+prompt|instructions?|full\s+prompt)",
    ]
]


class InputGuard:
    """
    Validates incoming user queries before they enter the RAG pipeline.

    Checks (in order):
    1. Toxicity — blocks harmful or hateful content
    2. Prompt injection — detects attempts to override system instructions
    """

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Returns (passed, reason).

        passed=True means the input is clean.
        If passed=False, reason is one of:
            "toxic_content"       → input triggered toxicity filter
            "injection_detected"  → input looks like a prompt injection attack
        """
        # Step 1: toxicity
        if self._is_toxic(text):
            return False, "toxic_content"

        # Step 2: injection
        if self._is_injection(text):
            return False, "injection_detected"

        return True, None

    @staticmethod
    def _is_toxic(text: str) -> bool:
        return any(p.search(text) for p in _TOXIC_PATTERNS)

    @staticmethod
    def _is_injection(text: str) -> bool:
        return any(p.search(text) for p in _INJECTION_PATTERNS)
