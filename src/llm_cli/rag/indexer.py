"""Build and persist a FAISS index alongside JSON metadata."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class FaissIndex:
    """FAISS IndexFlatIP (inner product = cosine for normalized vectors)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape == (len(chunks), self.dim), (
            f"Embeddings shape {embeddings.shape} doesn't match (N, {self.dim})"
        )
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top_k (chunk, score) pairs ordered by descending similarity."""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output_dir / "index.faiss"))

        metadata = [
            {**chunk.metadata, "text": chunk.text}
            for chunk in self.chunks
        ]
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: str | Path, dim: int) -> "FaissIndex":
        index_dir = Path(index_dir)
        obj = cls(dim)
        obj.index = faiss.read_index(str(index_dir / "index.faiss"))

        raw = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        for item in raw:
            text = item.pop("text")
            obj.chunks.append(Chunk(text=text, metadata=item))

        return obj
