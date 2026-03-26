from __future__ import annotations

from app.graph.state import RAGState
from app.services.vector_store import VectorStoreService
from app.services.bm25_index import BM25Index


def make_retrieve_node(
    vector_store: VectorStoreService,
    bm25_index: BM25Index,
    vector_top_k: int = 10,
    bm25_top_k: int = 10,
):
    """
    Step 6 — Hybrid retrieval.

    Runs dense (HNSW vector search) and sparse (BM25) retrieval in
    parallel-equivalent fashion (both use the same embedding/query).

    - Dense:  captures semantic similarity — good for paraphrased questions
    - Sparse: captures keyword/term overlap — good for abbreviations and exact terms

    Results are kept separate; fusion happens in the next node (RRF).
    """
    def retrieve_node(state: RAGState) -> dict:
        embedding = state["query_embedding"]
        query = state["query"]

        # Dense retrieval (Qdrant HNSW)
        vector_chunks = vector_store.dense_search(embedding, top_k=vector_top_k)

        # Sparse retrieval (BM25 in-memory)
        bm25_chunks = bm25_index.search(query, top_k=bm25_top_k)

        return {
            "vector_chunks": vector_chunks,
            "bm25_chunks": bm25_chunks,
        }

    return retrieve_node
