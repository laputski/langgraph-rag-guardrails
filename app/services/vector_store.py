from __future__ import annotations

from typing import Optional
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    SearchRequest,
)

from app.graph.state import ScoredChunk


class VectorStoreService:
    """Thin wrapper around Qdrant for dense vector search."""

    def __init__(self, url: str, collection: str, dim: int = 384) -> None:
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._dim = dim

    # ── Setup ─────────────────────────────────────────────────────────────────

    def create_collection_if_not_exists(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def drop_collection(self) -> None:
        self._client.delete_collection(self._collection)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        Each chunk dict must have keys:
            doc_id, title, chunk_index, text, source_file, embedding (list[float])
        """
        points = []
        for chunk in chunks:
            point_id = str(
                uuid5(NAMESPACE_DNS, f"{chunk['doc_id']}:{chunk['chunk_index']}")
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "source_file": chunk["source_file"],
                    },
                )
            )
        # Upload in batches of 100
        for i in range(0, len(points), 100):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + 100],
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def dense_search(
        self, embedding: list[float], top_k: int = 10
    ) -> list[ScoredChunk]:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        chunks = []
        for r in results:
            p = r.payload
            chunks.append(
                ScoredChunk(
                    doc_id=p["doc_id"],
                    title=p["title"],
                    chunk_index=p["chunk_index"],
                    text=p["text"],
                    source_file=p["source_file"],
                    score=r.score,
                )
            )
        return chunks

    def get_all_documents(self) -> list[ScoredChunk]:
        """Scroll through all stored chunks — used to build the BM25 index."""
        chunks: list[ScoredChunk] = []
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                p = r.payload
                chunks.append(
                    ScoredChunk(
                        doc_id=p["doc_id"],
                        title=p["title"],
                        chunk_index=p["chunk_index"],
                        text=p["text"],
                        source_file=p["source_file"],
                    )
                )
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
