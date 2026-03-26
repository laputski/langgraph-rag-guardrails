from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from ingest.chunker import MarkdownChunker


KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"


def _strip_frontmatter_text(text: str) -> str:
    """Remove YAML frontmatter delimiters from text body."""
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def load_and_chunk_all() -> list[dict]:
    """
    Walk knowledge_base/*.md, parse frontmatter, and chunk each document.

    Returns a list of chunk dicts ready for embedding + Qdrant upsert:
        {doc_id, title, chunk_index, text, source_file}
    """
    chunker = MarkdownChunker(max_tokens=400, overlap_tokens=50)
    all_chunks: list[dict] = []

    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No markdown files found in {KNOWLEDGE_BASE_DIR}. "
            "Did you run from the project root?"
        )

    for md_file in md_files:
        raw = md_file.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)

        title: str = post.metadata.get("title", md_file.stem.replace("_", " ").title())
        doc_id: str = md_file.stem
        body: str = post.content.strip()

        chunks = chunker.chunk(body)
        for chunk in chunks:
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "source_file": md_file.name,
                }
            )

    return all_chunks


async def run_ingest(
    vector_store: VectorStoreService,
    embedding_service: EmbeddingService,
    force_rebuild: bool = False,
) -> tuple[int, int]:
    """
    Load, chunk, embed, and upsert all knowledge base documents.

    Returns (chunks_indexed, num_documents).
    """
    if force_rebuild:
        try:
            vector_store.drop_collection()
        except Exception:
            pass
        vector_store.create_collection_if_not_exists()

    chunks = load_and_chunk_all()
    num_documents = len({c["doc_id"] for c in chunks})

    # Embed all chunks in a single batch call
    texts = [c["text"] for c in chunks]
    embeddings = embedding_service.embed_batch(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    vector_store.upsert_chunks(chunks)
    return len(chunks), num_documents
