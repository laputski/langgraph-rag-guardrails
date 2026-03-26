"""
CLI entry point for ingesting the knowledge base.

Usage:
    python -m ingest.run [--force-rebuild]
"""
from __future__ import annotations

import asyncio
import argparse
import time

from app.config import settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from ingest.loader import run_ingest


async def main(force_rebuild: bool) -> None:
    print("=" * 60)
    print("  Acme Corp HR — Knowledge Base Ingestion")
    print("=" * 60)

    print(f"\nLoading embedding model: {settings.embedding_model}")
    embedding_service = EmbeddingService(settings.embedding_model)
    print(f"✓ Embedding model loaded (dim={embedding_service.dim})")

    print(f"\nConnecting to Qdrant: {settings.qdrant_url}")
    vector_store = VectorStoreService(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=embedding_service.dim,
    )
    vector_store.create_collection_if_not_exists()
    print(f"✓ Qdrant connected (collection: {settings.qdrant_collection})")

    if force_rebuild:
        print("\n⚠ force-rebuild: dropping and recreating collection")

    print("\nIngesting documents...")
    t0 = time.time()
    chunks_indexed, documents = await run_ingest(
        vector_store=vector_store,
        embedding_service=embedding_service,
        force_rebuild=force_rebuild,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"  ✅ Ingestion complete in {elapsed:.1f}s")
    print(f"     Documents : {documents}")
    print(f"     Chunks    : {chunks_indexed}")
    print(f"     Collection: {settings.qdrant_collection}")
    print("=" * 60)
    print("\nYou can now start the server with: make dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base into Qdrant")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()
    asyncio.run(main(force_rebuild=args.force_rebuild))
