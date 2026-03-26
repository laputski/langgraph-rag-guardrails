from __future__ import annotations

from collections import defaultdict

from app.graph.state import RAGState, ScoredChunk
from app.services.reranker import RerankerService


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def rrf_fusion(
    vector_chunks: list[ScoredChunk],
    bm25_chunks: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Each document receives a combined score based on its rank in both lists.
    Documents that appear in both lists receive additive bonuses.
    Deduplication is performed by (doc_id, chunk_index).
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, ScoredChunk] = {}

    for rank, chunk in enumerate(vector_chunks, start=1):
        key = chunk.unique_key()
        scores[key] += _rrf_score(rank, k)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(bm25_chunks, start=1):
        key = chunk.unique_key()
        scores[key] += _rrf_score(rank, k)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        ScoredChunk(
            doc_id=chunk_map[k].doc_id,
            title=chunk_map[k].title,
            chunk_index=chunk_map[k].chunk_index,
            text=chunk_map[k].text,
            source_file=chunk_map[k].source_file,
            score=scores[k],
        )
        for k in sorted_keys
    ]


# ── Node factory ──────────────────────────────────────────────────────────────

def make_fuse_rerank_node(
    reranker: RerankerService,
    rerank_top_k: int = 5,
):
    """
    Step 7 — RRF fusion followed by cross-encoder reranking.

    1. Merge vector_chunks and bm25_chunks via RRF.
    2. Pass fused candidates to the cross-encoder reranker.
    3. Keep top-k results as reranked_chunks.
    """
    def fuse_rerank_node(state: RAGState) -> dict:
        fused = rrf_fusion(state["vector_chunks"], state["bm25_chunks"])
        reranked = reranker.rerank(state["query"], fused, top_k=rerank_top_k)

        return {
            "fused_chunks": fused,
            "reranked_chunks": reranked,
        }

    return fuse_rerank_node
