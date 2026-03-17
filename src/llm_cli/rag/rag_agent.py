"""RAG-агент: обёртка вокруг Agent + FaissIndex для ответов с/без контекста."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .chunker import Chunk
from .embedder import Embedder
from .indexer import FaissIndex

if TYPE_CHECKING:
    from ..agent import Agent

_PROJECT_ROOT = Path(__file__).parents[3]
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"

_RAG_SYSTEM_PROMPT = (
    "Ты ассистент, специализирующийся на курсе HuggingFace NLP Course. "
    "Тебе будут предоставлены фрагменты документации — используй их как основу для ответа, "
    "дополняя своими знаниями там, где источники неполны. "
    "Давай развёрнутые, полезные ответы. "
    "Технические термины (Trainer, AutoModel, from_pretrained, load_dataset, BPE, WordPiece, "
    "padding, truncation, attention_mask, fine-tuning, tokenizer и т.п.) пиши на английском. "
    "Объяснения и связующий текст — на русском языке."
)

_NO_RAG_SYSTEM_PROMPT = (
    "Ты ассистент по NLP и машинному обучению. "
    "Отвечай на русском языке, сохраняя технические термины на английском."
)


@dataclass
class RagAnswer:
    text: str
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    used_rag: bool = False

    @property
    def sources(self) -> list[str]:
        """Список уникальных источников (файлов)."""
        seen: set[str] = set()
        result: list[str] = []
        for chunk, _ in self.chunks:
            src = chunk.metadata.get("file", chunk.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                result.append(src)
        return result


class RagAgent:
    """Агент с двумя режимами: с RAG (поиск по FAISS + LLM) и без RAG (только LLM).

    Использует существующий Agent для обращений к LLM и FaissIndex для поиска.
    Каждый вызов ask() создаёт свежий Agent без истории, чтобы вопросы были
    независимы — это нужно для честного eval-сравнения.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        index_dir: Path | None = None,
        strategy: str = "structural",
        top_k: int = 5,
        temperature: float | None = 0.3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._index_dir = (index_dir or _INDEX_DIR) / strategy
        self._top_k = top_k
        self._temperature = temperature
        self._strategy = strategy

        self._embedder: Embedder | None = None
        self._index: FaissIndex | None = None

    def _ensure_index(self) -> None:
        if self._index is None:
            self._embedder = Embedder()
            self._index = FaissIndex.load(self._index_dir, dim=self._embedder.dim)

    def _translate_query(self, question: str, client: "OpenRouterClient") -> str:  # type: ignore[name-defined]
        """Перевести вопрос на английский для поиска в английском индексе.

        Использует быстрый LLM-запрос без истории. Если перевод не удался —
        возвращает оригинальный вопрос.
        """
        from ..models import ChatMessage

        # Если вопрос уже на английском (>50% ASCII-букв) — не переводим
        ascii_letters = sum(1 for c in question if c.isascii() and c.isalpha())
        total_letters = sum(1 for c in question if c.isalpha())
        if total_letters > 0 and ascii_letters / total_letters > 0.5:
            return question

        try:
            messages = [
                ChatMessage(role="system", content="Translate the following question to English. Return ONLY the translation, nothing else."),
                ChatMessage(role="user", content=question),
            ]
            translated = client.send(messages, model=self._model, temperature=0.0)
            return translated.strip() if translated.strip() else question
        except Exception:
            return question

    def _retrieve(self, question: str, client: "OpenRouterClient | None" = None) -> list[tuple[Chunk, float]]:  # type: ignore[name-defined]
        self._ensure_index()
        assert self._embedder is not None and self._index is not None
        # Переводим на английский для лучшего семантического поиска по английскому индексу
        search_query = self._translate_query(question, client) if client is not None else question
        q_emb = self._embedder.encode([search_query], show_progress=False)
        return self._index.search(q_emb, top_k=self._top_k)

    def _build_rag_prompt(self, question: str, chunks: list[tuple[Chunk, float]]) -> str:
        lines = [
            "Ниже приведены фрагменты документации HuggingFace NLP Course.",
            "Используй их как основу для ответа, дополняя своими знаниями.",
            "",
            "--- Фрагменты документации ---",
        ]
        for i, (chunk, score) in enumerate(chunks, 1):
            title = chunk.metadata.get("title", "")
            chapter = chunk.metadata.get("chapter", "")
            section = chunk.metadata.get("section", "")
            label_parts = [p for p in [chapter, title, section] if p]
            label = " · ".join(label_parts) if label_parts else "документация"
            lines.append(f"\n[{i}] {label} (relevance={score:.2f})")
            lines.append(chunk.text.strip())
        lines.append("\n--- Конец фрагментов ---")
        lines.append("")
        lines.append(f"Вопрос: {question}")
        lines.append("")
        lines.append(
            "Дай подробный ответ на русском языке (минимум 4–6 предложений), "
            "опираясь на информацию из источников выше. "
            "Обязательно упомяни конкретные технические термины и концепции из документации."
        )
        return "\n".join(lines)

    def ask(self, question: str, use_rag: bool = True) -> RagAnswer:
        """Задать вопрос агенту.

        Args:
            question: Вопрос пользователя.
            use_rag: Если True — ищет релевантные чанки и добавляет их в промпт.
                     Если False — запрашивает LLM напрямую без контекста.

        Returns:
            RagAnswer с текстом ответа и (при use_rag=True) использованными чанками.
        """
        from ..agent import Agent
        from ..api import OpenRouterClient

        system_prompt = _RAG_SYSTEM_PROMPT if use_rag else _NO_RAG_SYSTEM_PROMPT

        with OpenRouterClient(self._api_key) as client:
            agent = Agent(
                client=client,
                model=self._model,
                temperature=self._temperature,
                system_prompt=system_prompt,
                compression_enabled=False,
            )
            # Сбрасываем историю чтобы каждый вопрос был независимым
            agent._raw_history = [msg for msg in agent._raw_history if msg.role == "system"]

            if use_rag:
                chunks = self._retrieve(question, client=client)
                prompt = self._build_rag_prompt(question, chunks)
                text = agent.run(prompt)
                return RagAnswer(text=text, chunks=chunks, used_rag=True)
            else:
                text = agent.run(question)
                return RagAnswer(text=text, chunks=[], used_rag=False)
