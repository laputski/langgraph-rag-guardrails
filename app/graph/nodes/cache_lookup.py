from __future__ import annotations

from app.graph.state import RAGState
from app.services.semantic_cache import SemanticCache
from app.services.embedding import EmbeddingService


def make_cache_lookup_node(
    semantic_cache: SemanticCache,
    embedding_service: EmbeddingService,
):
    """
    Steps 3–4 — Semantic cache lookup.

    Embeds the query (same embedding is reused later for retrieval)
    and checks the Redis cache for a semantically similar previous answer.

    Step 3a (cache hit):  returns cached answer → pipeline short-circuits to streaming.
    Step 3b (cache miss): continues to embed_query → retrieval → …
    """
    async def cache_lookup_node(state: RAGState) -> dict:
        # Embed query (done here so the embedding is available for both cache
        # lookup and downstream retrieval)
        embedding = embedding_service.embed(state["query"])

        result = await semantic_cache.lookup(embedding, embedding_service)
        if result is not None:
            answer, sources, cache_key = result
            return {
                "query_embedding": embedding,
                "cache_hit": True,
                "cache_key": cache_key,
                "cached_answer": answer,
                "cached_sources": sources,
                "final_answer": answer,
                "sources": sources,
                "from_cache": True,
            }

        return {
            "query_embedding": embedding,
            "cache_hit": False,
            "cache_key": None,
            "cached_answer": None,
            "cached_sources": None,
        }

    return cache_lookup_node
