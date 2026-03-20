"""RAG-чат с историей диалога и памятью задачи (task state).

Ключевые отличия от run_interactive_chat():
  - История диалога накапливается между ходами (не сбрасывается)
  - Перед каждым LLM-вызовом в промпт инжектируется TaskMemory
  - Источники RAG выводятся принудительно при каждом ответе
  - Автоэкстракция сигналов задачи из сообщений пользователя
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk
from .rag_agent import RagAgent

_PROJECT_ROOT = Path(__file__).parents[3]
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"

# Максимум ходов диалога, передаваемых в контекст LLM (sliding window)
_DEFAULT_MAX_HISTORY = 20


# ─────────────────────────────────────────────────────────────────────────────
# Модели данных
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TaskMemory:
    """Память текущей задачи/сессии.

    Хранит цель диалога, уточнения пользователя, зафиксированные ограничения
    и термины. Инжектируется в каждый запрос к LLM как system-блок.
    """

    goal: str = ""
    clarifications: list[str] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0

    def is_empty(self) -> bool:
        return not self.goal and not self.clarifications and not self.constraints

    def build_block(self) -> str:
        """Сформировать system-блок для инъекции в промпт."""
        if self.is_empty():
            return ""
        lines = ["[ПАМЯТЬ ЗАДАЧИ]"]
        if self.goal:
            lines.append(f"Цель диалога: {self.goal}")
        if self.constraints:
            lines.append("Зафиксированные ограничения/термины:")
            for key, val in self.constraints.items():
                lines.append(f"  - {key}: {val}")
        if self.clarifications:
            lines.append("Уточнения пользователя:")
            for cl in self.clarifications[-5:]:
                lines.append(f"  - {cl}")
        lines.append(f"Ход диалога: {self.turn_count}")
        lines.append(
            "Используй эту память для сохранения контекста и не отклоняйся от цели."
        )
        return "\n".join(lines)


@dataclass
class RagMemoryAnswer:
    """Ответ чата с памятью."""

    answer: str
    sources: list[dict[str, str]]
    turn: int
    elapsed: float
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    rewritten_query: str | None = None
    unknown_due_to_low_relevance: bool = False

    @property
    def source_lines(self) -> list[str]:
        lines = []
        for ref in self.sources:
            src = ref.get("source", "?")
            section = ref.get("section", "")
            score = ref.get("score", "")
            chunk_id = ref.get("chunk_id", "")
            parts = [f"[{src}"]
            if section:
                parts.append(f" · {section}")
            parts.append("]")
            if score:
                parts.append(f"  score={score}")
            if chunk_id:
                parts.append(f"  id={chunk_id}")
            lines.append("".join(parts))
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Экстракция сигналов задачи из текста пользователя
# ─────────────────────────────────────────────────────────────────────────────

_GOAL_PATTERNS = [
    re.compile(r"(?i)(?:хочу|нужно|нужна|надо|цель|задача)[:\s]+(.{10,120})", re.DOTALL),
    re.compile(r"(?i)(?:изучаю|разбираюсь|изучить|понять|научиться)\s+(.{5,80})"),
    re.compile(r"(?i)(?:помоги|объясни|расскажи)\s+(?:мне\s+)?(.{5,80})"),
]

_CONSTRAINT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)только\s+(.{3,40}?)\s*(?:,|\.|$)"), "только"),
    (re.compile(r"(?i)без\s+(.{3,40}?)\s*(?:,|\.|$)"), "без"),
    (re.compile(r"(?i)(?:ограничение|условие)[:\s]+(.{5,80})"), "ограничение"),
    (re.compile(r"(?i)(?:термин|понятие)[:\s]+(.{3,40})"), "термин"),
]

_CLARIFICATION_PATTERNS = [
    re.compile(r"(?i)(?:точнее|уточняю|конкретно|имею в виду)[:\s]*(.{5,120})"),
    re.compile(r"(?i)(?:в частности|например|то есть)[:\s]+(.{5,120})"),
]


def _extract_goal(text: str) -> str | None:
    for pat in _GOAL_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = m.group(1).strip().rstrip(".,;")
            if len(snippet) >= 8:
                return snippet[:100]
    return None


def _extract_constraints(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pat, label in _CONSTRAINT_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1).strip().rstrip(".,;")
            if len(val) >= 3:
                key = f"{label}_{len(result)}" if label in result else label
                result[key] = val[:60]
    return result


def _extract_clarification(text: str) -> str | None:
    for pat in _CLARIFICATION_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = m.group(1).strip().rstrip(".,;")
            if len(snippet) >= 8:
                return snippet[:100]
    return None


def _build_source_refs(chunks: list[tuple[Chunk, float]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for chunk, score in chunks:
        refs.append({
            "source": str(chunk.metadata.get("file", chunk.metadata.get("source", "unknown"))),
            "section": str(chunk.metadata.get("section", "")),
            "chunk_id": str(chunk.metadata.get("chunk_id", "")),
            "score": f"{score:.3f}",
        })
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────


class RagMemoryChat:
    """Мини-чат: RAG + история диалога + память задачи.

    Хранит полную историю в `_history` (list[dict]) и использует sliding window
    при построении промпта, чтобы не превысить контекст.
    """

    def __init__(
        self,
        agent: RagAgent,
        max_history: int = _DEFAULT_MAX_HISTORY,
    ) -> None:
        self._agent = agent
        self._max_history = max_history
        self._history: list[dict[str, str]] = []
        self._task_memory = TaskMemory()

    @property
    def task_memory(self) -> TaskMemory:
        return self._task_memory

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def reset(self) -> None:
        """Сбросить историю и память задачи."""
        self._history.clear()
        self._task_memory = TaskMemory()

    # ─────────────────────────────────────────────────────────────────────────

    def chat(self, question: str) -> RagMemoryAnswer:
        """Отправить вопрос, получить ответ с RAG, источниками и обновлённой памятью."""
        from ..api import OpenRouterClient
        from ..models import ChatMessage

        self._task_memory.turn_count += 1

        # 1. Обновить память задачи
        self._extract_task_signals(question)

        # 2. RAG-поиск и вызов LLM
        t0 = time.perf_counter()
        self._agent._ensure_index()

        with OpenRouterClient(self._agent._api_key) as client:
            chunks, retrieval_stats = self._agent._retrieve(
                question,
                client=client,
                rewrite_enabled=True,
            )

            # 3. Сборка промпта с историей + task_memory + RAG-контекстом
            system_content = self._build_system_prompt()
            messages: list[ChatMessage] = [
                ChatMessage(role="system", content=system_content),
            ]

            # Sliding window по истории (передаём последние max_history ходов)
            history_window = self._history[-self._max_history:]
            for turn in history_window:
                messages.append(ChatMessage(role="user", content=turn["user"]))
                messages.append(ChatMessage(role="assistant", content=turn["assistant"]))

            # Текущий вопрос с RAG-контекстом
            user_prompt = self._build_rag_user_prompt(question, chunks)
            messages.append(ChatMessage(role="user", content=user_prompt))

            # 4. Вызов LLM
            answer_text = client.send(
                messages,
                model=self._agent._model,
                temperature=self._agent._temperature,
            )

        elapsed = time.perf_counter() - t0

        if not answer_text or not answer_text.strip():
            answer_text = "Не удалось получить ответ от модели."

        # 5. Сохранить в историю (сохраняем оригинальный вопрос, не промпт с RAG)
        self._history.append({
            "user": question,
            "assistant": answer_text.strip(),
        })

        rewritten = (
            retrieval_stats.rewritten_query
            if retrieval_stats.query_rewritten
            else None
        )

        return RagMemoryAnswer(
            answer=answer_text.strip(),
            sources=_build_source_refs(chunks),
            turn=self._task_memory.turn_count,
            elapsed=elapsed,
            chunks=chunks,
            rewritten_query=rewritten,
            unknown_due_to_low_relevance=retrieval_stats.fallback_used,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        base = (
            "Ты ассистент, специализирующийся на курсе HuggingFace NLP Course. "
            "Ты ведёшь многоходовой диалог — помни всю историю беседы. "
            "При каждом ответе опирайся на предоставленные фрагменты документации. "
            "Технические термины (Trainer, AutoModel, from_pretrained, load_dataset, "
            "BPE, WordPiece, padding, truncation, attention_mask, fine-tuning, tokenizer) "
            "пиши на английском. Объяснения — на русском языке."
        )
        memory_block = self._task_memory.build_block()
        if memory_block:
            return f"{base}\n\n{memory_block}"
        return base

    def _build_rag_user_prompt(
        self, question: str, chunks: list[tuple[Chunk, float]]
    ) -> str:
        if not chunks:
            return question

        lines = [
            "--- Фрагменты документации HuggingFace NLP Course ---",
        ]
        for i, (chunk, score) in enumerate(chunks, 1):
            title = chunk.metadata.get("title", "")
            chapter = chunk.metadata.get("chapter", "")
            section = chunk.metadata.get("section", "")
            label_parts = [p for p in [chapter, title, section] if p]
            label = " · ".join(label_parts) if label_parts else "документация"
            lines.append(f"\n[{i}] {label}  (relevance={score:.2f})")
            lines.append(chunk.text.strip())
        lines.append("\n--- Конец фрагментов ---\n")
        lines.append(f"Вопрос: {question}")
        lines.append(
            "\nДай развёрнутый ответ на русском языке, опираясь на фрагменты выше. "
            "Если вопрос связан с предыдущими ходами диалога — учти историю беседы."
        )
        return "\n".join(lines)

    def _extract_task_signals(self, text: str) -> None:
        """Обновить память задачи на основе текста пользователя."""
        if not self._task_memory.goal:
            goal = _extract_goal(text)
            if goal:
                self._task_memory.goal = goal

        new_constraints = _extract_constraints(text)
        self._task_memory.constraints.update(new_constraints)

        clarification = _extract_clarification(text)
        if clarification and clarification not in self._task_memory.clarifications:
            self._task_memory.clarifications.append(clarification)
