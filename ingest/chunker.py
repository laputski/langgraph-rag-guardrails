from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int


def _count_tokens(text: str) -> int:
    """Rough token count: split on whitespace (1 token ≈ 1 word for English)."""
    return len(text.split())


class MarkdownChunker:
    """
    Splits a markdown document into overlapping chunks.

    Strategy:
    - Split on double newlines (paragraph boundaries).
    - Accumulate paragraphs until the token limit is reached.
    - When the limit is reached, save the chunk and start a new one
      with an overlap of the last `overlap_tokens` tokens from the previous chunk.

    This preserves sentence continuity across chunk boundaries.
    """

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> list[Chunk]:
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            # If a single paragraph exceeds max_tokens, split it by sentences
            if para_tokens > self._max_tokens:
                sub_chunks = self._split_long_paragraph(para)
                for sub in sub_chunks:
                    chunks.append(Chunk(text=sub, chunk_index=chunk_index))
                    chunk_index += 1
                current = []
                current_tokens = 0
                continue

            if current_tokens + para_tokens > self._max_tokens and current:
                # Save current chunk
                chunk_text = "\n\n".join(current)
                chunks.append(Chunk(text=chunk_text, chunk_index=chunk_index))
                chunk_index += 1

                # Start next chunk with overlap
                current, current_tokens = self._apply_overlap(current)

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(Chunk(text="\n\n".join(current), chunk_index=chunk_index))

        return chunks

    def _apply_overlap(
        self, paragraphs: list[str]
    ) -> tuple[list[str], int]:
        """Keep the last N tokens from the current chunk as overlap."""
        overlap_paras: list[str] = []
        token_count = 0
        for para in reversed(paragraphs):
            n = _count_tokens(para)
            if token_count + n > self._overlap_tokens:
                break
            overlap_paras.insert(0, para)
            token_count += n
        return overlap_paras, token_count

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split a long paragraph into sentence-level sub-chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", para)
        sub_chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            n = _count_tokens(sentence)
            if current_tokens + n > self._max_tokens and current_sentences:
                sub_chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            current_sentences.append(sentence)
            current_tokens += n

        if current_sentences:
            sub_chunks.append(" ".join(current_sentences))

        return sub_chunks
