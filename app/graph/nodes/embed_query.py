from __future__ import annotations

from app.graph.state import RAGState
from app.services.embedding import EmbeddingService


def make_embed_query_node(embedding_service: EmbeddingService):
    """
    Step 5 — Create query embedding.

    Embeds the (possibly rewritten) query using the local sentence-transformers
    model. The embedding is reused by both dense retrieval and cache lookup.
    """
    def embed_query_node(state: RAGState) -> dict:
        embedding = embedding_service.embed(state["query"])
        return {"query_embedding": embedding}

    return embed_query_node
