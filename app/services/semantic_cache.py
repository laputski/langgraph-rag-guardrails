from __future__ import annotations

import json
import uuid
from typing import Optional

import redis.asyncio as aioredis

from app.services.embedding import EmbeddingService


class SemanticCache:
    """
    Redis-backed semantic cache using cosine similarity on query embeddings.

    Storage format — Redis hash at key `cache:{uuid}`:
        embedding_json : JSON-serialized list[float]
        answer         : str
        sources_json   : JSON-serialized list[str]

    Lookup algorithm:
        1. Embed the incoming query.
        2. Scan all `cache:*` keys (acceptable for demo; use Redis Stack HNSW
           for production with thousands of cached entries).
        3. Compute cosine similarity vs each stored embedding.
        4. Return the best match if similarity >= threshold.

    TTL is applied per entry (default 3600 s).
    """

    KEY_PREFIX = "cache:"

    def __init__(
        self,
        redis_url: str,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ) -> None:
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds

    # ── Public API ────────────────────────────────────────────────────────────

    async def lookup(
        self,
        query_embedding: list[float],
        embedding_service: EmbeddingService,
    ) -> Optional[tuple[str, list[str], str]]:
        """
        Returns (answer, sources, cache_key) if a similar entry is found,
        else None.
        """
        keys = await self._scan_keys()
        best_sim = -1.0
        best_match: Optional[tuple[str, list[str], str]] = None

        for key in keys:
            data = await self._redis.hgetall(key)
            if not data:
                continue
            stored_emb: list[float] = json.loads(data["embedding_json"])
            sim = EmbeddingService.cosine_similarity(query_embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_match = (
                    data["answer"],
                    json.loads(data["sources_json"]),
                    key,
                )

        if best_sim >= self._threshold and best_match is not None:
            return best_match
        return None

    async def store(
        self,
        query_embedding: list[float],
        answer: str,
        sources: list[str],
    ) -> str:
        """Store an answer in the cache. Returns the cache key."""
        key = f"{self.KEY_PREFIX}{uuid.uuid4().hex}"
        mapping = {
            "embedding_json": json.dumps(query_embedding),
            "answer": answer,
            "sources_json": json.dumps(sources),
        }
        await self._redis.hset(key, mapping=mapping)
        await self._redis.expire(key, self._ttl)
        return key

    async def health(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._redis.aclose()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _scan_keys(self) -> list[str]:
        keys: list[str] = []
        cursor = 0
        while True:
            cursor, batch = await self._redis.scan(
                cursor=cursor, match=f"{self.KEY_PREFIX}*", count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break
        return keys
