from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wraps a local sentence-transformers model for query and document embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Model is downloaded on first run (~90MB), then cached in ~/.cache/huggingface/
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a normalized float list."""
        vec: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts efficiently in a single forward pass."""
        vecs: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two L2-normalized vectors (dot product suffices)."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return float(np.dot(va, vb))
