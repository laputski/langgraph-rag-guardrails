from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.graph.state import ScoredChunk


class RerankerService:
    """
    Cross-encoder reranker.

    Unlike bi-encoder embeddings (which encode query and document independently),
    a cross-encoder scores the (query, chunk) pair jointly, giving significantly
    more accurate relevance scores at the cost of O(n) inference calls.

    Applied only to the Top-K candidates after RRF fusion — not the full corpus.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        # Model is ~68MB, downloaded once to ~/.cache/huggingface/
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """
        Score each (query, chunk) pair and return the top-k chunks
        sorted by cross-encoder score descending.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            zip(scores, chunks), key=lambda x: x[0], reverse=True
        )[:top_k]

        return [
            ScoredChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                source_file=chunk.source_file,
                score=score,
            )
            for score, chunk in ranked
        ]
