"""RAG-агент: обёртка вокруг Agent + FaissIndex для ответов с/без контекста."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .chunker import Chunk
from .embedder import Embedder
from .indexer import FaissIndex
from .relevance import PostRetrievalMode, PostRetrievalStats, apply_post_retrieval
from ..models import OPENROUTER_URL

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


_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


@dataclass
class RagAnswer:
    text: str
    answer: str = ""
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    used_rag: bool = False
    retrieval_stats: PostRetrievalStats | None = None
    unknown_due_to_low_relevance: bool = False
    relevance_threshold: float | None = None

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

    @property
    def source_refs(self) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        for chunk, score in self.chunks:
            refs.append(
                {
                    "source": str(chunk.metadata.get("file", chunk.metadata.get("source", "unknown"))),
                    "section": str(chunk.metadata.get("section", "")),
                    "chunk_id": str(chunk.metadata.get("chunk_id", "")),
                    "score": f"{score:.3f}",
                }
            )
        return refs

    @property
    def citations(self) -> list[str]:
        quotes: list[str] = []
        for chunk, _ in self.chunks:
            snippet = " ".join(chunk.text.strip().split())
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            quotes.append(snippet)
        return quotes


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text) if len(m.group(0)) >= 3}


def _answer_supported_by_citations(answer: str, citations: list[str]) -> bool:
    if not answer.strip() or not citations:
        return False
    a_tokens = _tokenize(answer)
    if not a_tokens:
        return False
    c_tokens = _tokenize(" ".join(citations))
    if not c_tokens:
        return False
    overlap = len(a_tokens & c_tokens) / len(a_tokens)
    return overlap >= 0.1


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
        top_k_before: int | None = None,
        top_k_after: int | None = None,
        min_similarity: float = 0.0,
        post_retrieval_mode: PostRetrievalMode = "off",
        rewrite_enabled: bool = True,
        temperature: float | None = 0.3,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or OPENROUTER_URL
        self._index_dir = (index_dir or _INDEX_DIR) / strategy
        resolved_top_k_before = top_k_before if top_k_before is not None else top_k
        resolved_top_k_after = top_k_after if top_k_after is not None else resolved_top_k_before
        if resolved_top_k_before <= 0:
            raise ValueError("top_k_before must be > 0")
        if resolved_top_k_after <= 0:
            raise ValueError("top_k_after must be > 0")
        if post_retrieval_mode not in {"off", "threshold", "rerank"}:
            raise ValueError("post_retrieval_mode must be one of: off, threshold, rerank")

        self._top_k_before = resolved_top_k_before
        self._top_k_after = resolved_top_k_after
        self._min_similarity = min_similarity
        self._post_retrieval_mode: PostRetrievalMode = post_retrieval_mode
        self._rewrite_enabled = rewrite_enabled
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

    def _retrieve(
        self,
        question: str,
        client: "OpenRouterClient | None" = None,  # type: ignore[name-defined]
        *,
        rewrite_enabled: bool | None = None,
        post_retrieval_mode: PostRetrievalMode | None = None,
        top_k_before: int | None = None,
        top_k_after: int | None = None,
        min_similarity: float | None = None,
    ) -> tuple[list[tuple[Chunk, float]], PostRetrievalStats]:
        self._ensure_index()
        assert self._embedder is not None and self._index is not None
        rewrite = self._rewrite_enabled if rewrite_enabled is None else rewrite_enabled
        mode = self._post_retrieval_mode if post_retrieval_mode is None else post_retrieval_mode
        retrieval_top_k = self._top_k_before if top_k_before is None else top_k_before
        selected_top_k = self._top_k_after if top_k_after is None else top_k_after
        threshold = self._min_similarity if min_similarity is None else min_similarity

        # Переводим на английский для лучшего семантического поиска по английскому индексу
        search_query = (
            self._translate_query(question, client) if (rewrite and client is not None) else question
        )
        q_emb = self._embedder.encode([search_query], show_progress=False)
        raw_chunks = self._index.search(q_emb, top_k=retrieval_top_k)
        selected_chunks, stats = apply_post_retrieval(
            raw_chunks,
            query=search_query,
            mode=mode,
            top_k_before=retrieval_top_k,
            top_k_after=selected_top_k,
            min_similarity=threshold,
        )
        stats.query_rewritten = search_query.strip() != question.strip()
        stats.rewritten_query = search_query if stats.query_rewritten else None
        return selected_chunks, stats

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
            "Обязательно упомяни конкретные технические термины и концепции из документации. "
            "Верни ТОЛЬКО основной ответ без секций, списков источников и цитат."
        )
        return "\n".join(lines)

    def _format_sources(self, chunks: list[tuple[Chunk, float]]) -> list[str]:
        lines: list[str] = []
        for chunk, score in chunks:
            source = chunk.metadata.get("file", chunk.metadata.get("source", "unknown"))
            section = chunk.metadata.get("section", "")
            chunk_id = chunk.metadata.get("chunk_id", "")
            lines.append(
                f"- source: {source} | section: {section or '-'} | chunk_id: {chunk_id or '-'} | score: {score:.3f}"
            )
        return lines

    def _format_citations(self, chunks: list[tuple[Chunk, float]]) -> list[str]:
        lines: list[str] = []
        for chunk, _ in chunks:
            chunk_id = chunk.metadata.get("chunk_id", "-")
            snippet = " ".join(chunk.text.strip().split())
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            lines.append(f'- [{chunk_id}] "{snippet}"')
        return lines

    def _compose_structured_output(
        self,
        *,
        answer: str,
        chunks: list[tuple[Chunk, float]],
        unknown_due_to_low_relevance: bool,
        relevance_threshold: float | None,
    ) -> str:
        lines = ["Ответ:", answer.strip() or "Не удалось сформировать ответ.", "", "Источники:"]
        if chunks:
            lines.extend(self._format_sources(chunks))
        else:
            threshold_text = f"{relevance_threshold:.3f}" if relevance_threshold is not None else "n/a"
            lines.append(f"- нет релевантных источников (ниже порога relevance={threshold_text})")

        lines.extend(["", "Цитаты:"])
        if chunks:
            lines.extend(self._format_citations(chunks))
        else:
            lines.append("- нет цитат: релевантные чанки не найдены")

        if unknown_due_to_low_relevance:
            lines.extend(
                [
                    "",
                    "Комментарий:",
                    "Найденный контекст недостаточно релевантен, поэтому ответ дан в режиме 'не знаю'.",
                ]
            )
        return "\n".join(lines)

    def ask(
        self,
        question: str,
        use_rag: bool = True,
        *,
        rewrite_enabled: bool | None = None,
        post_retrieval_mode: PostRetrievalMode | None = None,
        top_k_before: int | None = None,
        top_k_after: int | None = None,
        min_similarity: float | None = None,
    ) -> RagAnswer:
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

        with OpenRouterClient(self._api_key, base_url=self._base_url) as client:
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
                chunks, retrieval_stats = self._retrieve(
                    question,
                    client=client,
                    rewrite_enabled=rewrite_enabled,
                    post_retrieval_mode=post_retrieval_mode,
                    top_k_before=top_k_before,
                    top_k_after=top_k_after,
                    min_similarity=min_similarity,
                )
                effective_threshold = self._min_similarity if min_similarity is None else min_similarity
                effective_mode = self._post_retrieval_mode if post_retrieval_mode is None else post_retrieval_mode

                low_relevance = (
                    effective_mode in {"threshold", "rerank"}
                    and effective_threshold > 0.0
                    and retrieval_stats.fallback_used
                )

                if low_relevance:
                    answer_text = (
                        "Не знаю: в найденном контексте нет достаточно релевантной информации. "
                        "Уточните вопрос: добавьте термин, раздел курса или более конкретный пример."
                    )
                    text = self._compose_structured_output(
                        answer=answer_text,
                        chunks=chunks,
                        unknown_due_to_low_relevance=True,
                        relevance_threshold=effective_threshold,
                    )
                    return RagAnswer(
                        text=text,
                        answer=answer_text,
                        chunks=chunks,
                        used_rag=True,
                        retrieval_stats=retrieval_stats,
                        unknown_due_to_low_relevance=True,
                        relevance_threshold=effective_threshold,
                    )

                prompt = self._build_rag_prompt(question, chunks)
                answer_text = agent.run(prompt).strip()

                # Мягкая проверка: если связь с цитатами низкая, оставляем ответ,
                # но добавляем предупреждение в конец.
                if not _answer_supported_by_citations(answer_text, [c.text for c, _ in chunks]):
                    answer_text = (
                        f"{answer_text}\n\n"
                        "Примечание: часть утверждений может выходить за пределы предоставленных цитат; "
                        "для надёжности уточните вопрос."
                    ).strip()

                text = self._compose_structured_output(
                    answer=answer_text,
                    chunks=chunks,
                    unknown_due_to_low_relevance=low_relevance,
                    relevance_threshold=effective_threshold if effective_threshold > 0 else None,
                )
                return RagAnswer(
                    text=text,
                    answer=answer_text,
                    chunks=chunks,
                    used_rag=True,
                    retrieval_stats=retrieval_stats,
                    unknown_due_to_low_relevance=low_relevance,
                    relevance_threshold=effective_threshold if effective_threshold > 0 else None,
                )
            else:
                text = agent.run(question)
                return RagAnswer(text=text, answer=text, chunks=[], used_rag=False)
