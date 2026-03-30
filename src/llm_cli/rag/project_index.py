"""RAG-индекс по документации проекта: README.md + docs/."""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "project"
_EMBEDDER_DIM = 384  # all-MiniLM-L6-v2


class ProjectRagIndex:
    """FAISS-индекс, построенный по README.md и папке docs/ проекта.

    При первом вызове search() автоматически строит или загружает индекс.
    Повторные вызовы используют кэшированный объект.
    """

    def __init__(self, project_root: Path | None = None, index_dir: Path | None = None) -> None:
        self._root = project_root or _PROJECT_ROOT
        self._index_dir = index_dir or _INDEX_DIR
        self._faiss_index = None  # лениво инициализируется
        self._embedder = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Вернуть top_k наиболее релевантных текстовых фрагментов."""
        self._ensure_loaded()
        assert self._faiss_index is not None
        assert self._embedder is not None

        q_emb = self._embedder.encode([query], show_progress=False)
        results = self._faiss_index.search(q_emb[0], top_k=top_k)
        return [chunk.text for chunk, _score in results]

    def rebuild(self) -> None:
        """Принудительно пересобрать индекс из исходных документов."""
        self._build_index()

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._faiss_index is not None:
            return
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()

    def _index_exists(self) -> bool:
        return (self._index_dir / "index.faiss").exists() and (
            self._index_dir / "metadata.json"
        ).exists()

    def _get_embedder(self):
        if self._embedder is None:
            from .embedder import Embedder
            self._embedder = Embedder(silent=True)
        return self._embedder

    def _load_index(self) -> None:
        from .indexer import FaissIndex
        self._get_embedder()
        self._faiss_index = FaissIndex.load(self._index_dir, dim=self._embedder.dim)

    def _build_index(self) -> None:
        from .chunker import StructuralChunker
        from .embedder import Embedder
        from .indexer import FaissIndex
        from .loader import Document

        self._embedder = Embedder(silent=True)
        docs = self._collect_documents()

        chunker = StructuralChunker(max_section_chars=1200)
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk(doc))

        if not all_chunks:
            # Нет документов — создаём пустой индекс, чтобы не падать
            self._faiss_index = FaissIndex(dim=self._embedder.dim)
            return

        texts = [c.text for c in all_chunks]
        embeddings = self._embedder.encode(texts, show_progress=False)

        faiss_idx = FaissIndex(dim=self._embedder.dim)
        faiss_idx.add(all_chunks, embeddings)
        faiss_idx.save(self._index_dir)
        self._faiss_index = faiss_idx

    def _collect_documents(self) -> list:
        """Собрать документы: README.md из корня + все *.md/*.mdx из docs/."""
        from .loader import Document

        docs: list[Document] = []

        # README.md в корне проекта
        readme = self._root / "README.md"
        if readme.exists():
            raw = readme.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(
                text=_clean_markdown(raw),
                metadata={
                    "source": "project_readme",
                    "file": "README.md",
                    "chapter": "root",
                    "title": "README",
                },
            ))

        # docs/ папка в корне проекта (если есть)
        docs_dir = self._root / "docs"
        if docs_dir.is_dir():
            from .loader import load_documents
            extra = load_documents(docs_dir)
            for doc in extra:
                doc.metadata["source"] = "project_docs"
            docs.extend(extra)

        return docs


def _clean_markdown(text: str) -> str:
    """Минимальная очистка markdown: убрать HTML-теги и лишние пробелы."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
