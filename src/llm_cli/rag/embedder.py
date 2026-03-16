"""Generate embeddings using a local sentence-transformers model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "all-MiniLM-L6-v2"
_FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Wrapper around SentenceTransformer for batched embedding generation."""

    def __init__(self, model_path: str | Path | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        resolved = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        # Use local path if it contains model files, otherwise fall back to HF hub
        if resolved.exists() and any(resolved.iterdir()):
            model_source = str(resolved)
        else:
            model_source = _FALLBACK_MODEL_NAME

        self.model = SentenceTransformer(model_source)
        self.batch_size = batch_size
        self.dim: int = self.model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into normalized float32 embeddings.
        Returns array of shape (N, dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)  # type: ignore[union-attr]
