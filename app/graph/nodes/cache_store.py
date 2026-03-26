from __future__ import annotations

from app.graph.state import RAGState
from app.services.semantic_cache import SemanticCache


def make_cache_store_node(semantic_cache: SemanticCache):
    """
    Step 11 — Save answer to the semantic cache with TTL.

    Called only after a successful guardrail check.
    Stores (query_embedding, answer, sources) in Redis so that future
    semantically similar queries can receive a cached response.
    """
    async def cache_store_node(state: RAGState) -> dict:
        embedding = state.get("query_embedding")
        answer = state.get("final_answer") or ""
        sources = state.get("sources", [])

        if embedding and answer:
            await semantic_cache.store(
                query_embedding=embedding,
                answer=answer,
                sources=sources,
            )

        return {}

    return cache_store_node
