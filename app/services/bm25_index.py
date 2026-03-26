from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from app.graph.state import ScoredChunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """
    In-memory BM25 index built from Qdrant documents at application startup.

    Trade-off note: rebuilding from Qdrant on every restart takes ~100ms for
    150 chunks. For production with 100K+ chunks, serialize the index with
    pickle/joblib and persist it alongside Qdrant.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[ScoredChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[ScoredChunk]) -> None:
        """Tokenize all chunks and build the BM25 index."""
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """
        Return top-k chunks by BM25 score.
        Scores are normalized to [0, 1] by dividing by the maximum score
        so they can be consumed consistently by RRF fusion.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)

        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        scored = sorted(
            enumerate(raw_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, raw_score in scored:
            if raw_score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                ScoredChunk(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    score=raw_score / max_score,
                )
            )
        return results

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0
