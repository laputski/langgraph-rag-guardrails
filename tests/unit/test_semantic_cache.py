"""Unit tests for semantic cache similarity logic (no real Redis required)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.semantic_cache import SemanticCache
from app.services.embedding import EmbeddingService


def _make_embedding(value: float, dim: int = 384) -> list[float]:
    """Create a normalized-ish embedding with a single dominant direction."""
    import math
    vec = [0.0] * dim
    vec[0] = value
    vec[1] = math.sqrt(1.0 - value ** 2) if abs(value) <= 1.0 else 0.0
    return vec


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(EmbeddingService.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(EmbeddingService.cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(EmbeddingService.cosine_similarity(a, b) + 1.0) < 1e-6

    def test_similar_vectors(self):
        import math
        a = [1.0, 0.0]
        angle = 0.1  # radians — very similar
        b = [math.cos(angle), math.sin(angle)]
        sim = EmbeddingService.cosine_similarity(a, b)
        assert sim > 0.99


class TestSemanticCacheLookup:
    """Tests the cosine-similarity threshold logic without real Redis."""

    def _make_cache(self, threshold: float = 0.92) -> SemanticCache:
        cache = SemanticCache.__new__(SemanticCache)
        cache._threshold = threshold
        cache._ttl = 3600
        cache._redis = MagicMock()
        return cache

    @pytest.mark.asyncio
    async def test_cache_hit_above_threshold(self):
        cache = self._make_cache(threshold=0.92)
        embedding_service = MagicMock(spec=EmbeddingService)

        stored_emb = [1.0, 0.0, 0.0]
        query_emb = [0.99, 0.14, 0.0]   # cosine sim ≈ 0.99 > 0.92

        # Mock Redis scan and hgetall
        cache._redis.scan = AsyncMock(return_value=(0, ["cache:abc"]))
        cache._redis.hgetall = AsyncMock(return_value={
            "embedding_json": json.dumps(stored_emb),
            "answer": "You get 15 PTO days",
            "sources_json": json.dumps(["PTO Policy"]),
        })

        with patch.object(EmbeddingService, "cosine_similarity", return_value=0.97):
            result = await cache.lookup(query_emb, embedding_service)

        assert result is not None
        answer, sources, key = result
        assert answer == "You get 15 PTO days"
        assert sources == ["PTO Policy"]

    @pytest.mark.asyncio
    async def test_cache_miss_below_threshold(self):
        cache = self._make_cache(threshold=0.92)
        embedding_service = MagicMock(spec=EmbeddingService)

        stored_emb = [1.0, 0.0, 0.0]
        query_emb = [0.5, 0.866, 0.0]   # cosine sim = 0.5 < 0.92

        cache._redis.scan = AsyncMock(return_value=(0, ["cache:abc"]))
        cache._redis.hgetall = AsyncMock(return_value={
            "embedding_json": json.dumps(stored_emb),
            "answer": "Some other answer",
            "sources_json": json.dumps([]),
        })

        with patch.object(EmbeddingService, "cosine_similarity", return_value=0.5):
            result = await cache.lookup(query_emb, embedding_service)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_cache_returns_none(self):
        cache = self._make_cache()
        embedding_service = MagicMock(spec=EmbeddingService)

        cache._redis.scan = AsyncMock(return_value=(0, []))  # no keys

        result = await cache.lookup([0.1, 0.2], embedding_service)
        assert result is None
