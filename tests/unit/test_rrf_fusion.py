"""Unit tests for RRF fusion logic."""
from __future__ import annotations

import pytest

from app.graph.nodes.fuse_rerank import rrf_fusion, _rrf_score
from app.graph.state import ScoredChunk


def _chunk(doc_id: str, chunk_index: int = 0, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        doc_id=doc_id,
        title=doc_id.replace("_", " ").title(),
        chunk_index=chunk_index,
        text=f"Text of {doc_id}",
        source_file=f"{doc_id}.md",
        score=score,
    )


class TestRRFScore:
    def test_rank_1_with_default_k(self):
        score = _rrf_score(rank=1, k=60)
        assert abs(score - 1 / 61) < 1e-9

    def test_higher_rank_lower_score(self):
        assert _rrf_score(1) > _rrf_score(5) > _rrf_score(10)

    def test_score_is_bounded(self):
        for rank in range(1, 100):
            s = _rrf_score(rank)
            assert 0 < s <= 1.0 / 61


class TestRRFFusion:
    def test_empty_inputs_return_empty(self):
        result = rrf_fusion([], [])
        assert result == []

    def test_single_list(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = rrf_fusion(chunks, [])
        assert len(result) == 3
        # Order should be preserved (a=rank1, b=rank2, c=rank3)
        assert result[0].doc_id == "a"
        assert result[1].doc_id == "b"

    def test_deduplication(self):
        """A chunk appearing in both lists should appear only once in output."""
        vector = [_chunk("a"), _chunk("b"), _chunk("c")]
        bm25 = [_chunk("c"), _chunk("a"), _chunk("d")]
        result = rrf_fusion(vector, bm25)

        doc_ids = [r.doc_id for r in result]
        # No duplicates
        assert len(doc_ids) == len(set(doc_ids))
        # "a" and "c" appear in both lists → should score higher than "b" and "d"
        assert doc_ids.index("a") < doc_ids.index("b")
        assert doc_ids.index("c") < doc_ids.index("b")

    def test_scores_are_combined(self):
        """A doc in both lists should have a higher score than one in only one list."""
        chunk_in_both = _chunk("shared")
        chunk_only_vector = _chunk("only_vector")
        chunk_only_bm25 = _chunk("only_bm25")

        vector = [chunk_in_both, chunk_only_vector]
        bm25 = [chunk_in_both, chunk_only_bm25]

        result = rrf_fusion(vector, bm25)
        scores = {r.doc_id: r.score for r in result}

        assert scores["shared"] > scores["only_vector"]
        assert scores["shared"] > scores["only_bm25"]

    def test_multi_chunk_same_doc_not_deduplicated(self):
        """Different chunks of the same doc have different unique_keys and are kept separate."""
        chunk_0 = _chunk("doc_a", chunk_index=0)
        chunk_1 = _chunk("doc_a", chunk_index=1)
        result = rrf_fusion([chunk_0, chunk_1], [])
        assert len(result) == 2

    def test_result_sorted_by_score_descending(self):
        vector = [_chunk("a"), _chunk("b"), _chunk("c")]
        bm25 = [_chunk("b"), _chunk("a"), _chunk("c")]
        result = rrf_fusion(vector, bm25)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
