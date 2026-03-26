from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict


@dataclass
class ScoredChunk:
    """A document chunk with a retrieval score."""
    doc_id: str
    title: str
    chunk_index: int
    text: str
    source_file: str
    score: float = 0.0

    def unique_key(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"


class RAGState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    user_id: Optional[str]
    request_id: str
    prompt_version: str

    # ── Input Guardrails (Step 2) ─────────────────────────────────────────────
    input_guard_passed: bool
    input_guard_reason: Optional[str]   # "toxic_content" | "injection_detected"

    # ── Semantic Cache (Steps 3–4) ────────────────────────────────────────────
    query_embedding: Optional[list[float]]
    cache_hit: bool
    cache_key: Optional[str]            # Redis key of the matched entry
    cached_answer: Optional[str]
    cached_sources: Optional[list[str]]

    # ── Retrieval (Steps 5–6) ─────────────────────────────────────────────────
    vector_chunks: list[ScoredChunk]    # from HNSW dense search
    bm25_chunks: list[ScoredChunk]      # from BM25 sparse search

    # ── Fusion + Rerank (Step 7) ──────────────────────────────────────────────
    fused_chunks: list[ScoredChunk]     # after RRF
    reranked_chunks: list[ScoredChunk]  # final top-k after cross-encoder

    # ── Prompt (Step 8) ───────────────────────────────────────────────────────
    context_text: Optional[str]         # formatted context string passed to LLM

    # ── LLM Generation (Step 9) ──────────────────────────────────────────────
    llm_answer: Optional[str]
    llm_retry_count: int
    llm_model_used: str

    # ── Output Guardrails (Step 10) ───────────────────────────────────────────
    output_guard_passed: bool
    output_guard_reason: Optional[str]  # "hallucination" | "toxic_output"

    # ── Final ─────────────────────────────────────────────────────────────────
    final_answer: Optional[str]
    sources: list[str]
    from_cache: bool
    error: Optional[str]
    error_code: Optional[str]           # "content_filtered" | "injection_detected" | "max_retries_exceeded"


def initial_state(
    query: str,
    request_id: str,
    user_id: Optional[str] = None,
    prompt_version: str = "v2",
) -> RAGState:
    """Create a fresh RAGState for a new request."""
    return RAGState(
        query=query,
        user_id=user_id,
        request_id=request_id,
        prompt_version=prompt_version,
        input_guard_passed=False,
        input_guard_reason=None,
        query_embedding=None,
        cache_hit=False,
        cache_key=None,
        cached_answer=None,
        cached_sources=None,
        vector_chunks=[],
        bm25_chunks=[],
        fused_chunks=[],
        reranked_chunks=[],
        context_text=None,
        llm_answer=None,
        llm_retry_count=0,
        llm_model_used="",
        output_guard_passed=False,
        output_guard_reason=None,
        final_answer=None,
        sources=[],
        from_cache=False,
        error=None,
        error_code=None,
    )
