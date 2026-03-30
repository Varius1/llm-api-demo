"""Ассистент разработчика: отвечает на вопросы о проекте с помощью RAG + git-контекста."""

from __future__ import annotations

import subprocess
from functools import cached_property
from pathlib import Path

from .api import OpenRouterClient
from .config import AppConfig
from .models import OPENROUTER_URL

_PROJECT_ROOT = Path(__file__).parents[2]

_SYSTEM_PROMPT_TEMPLATE = """\
Ты — ассистент разработчика проекта llm-api-demo.
Ты помогаешь разработчикам разобраться в структуре, архитектуре и коде проекта.

Текущий git-контекст:
{git_info}

Документация проекта (найденные фрагменты по теме вопроса):
---
{doc_context}
---

Отвечай конкретно и по делу. Если информации нет в контексте — честно скажи об этом.
Используй Markdown-форматирование в ответах.
"""


class DevAssistant:
    """Ассистент разработчика.

    Использует RAG-индекс по README + docs/ и вызывает git для получения
    текущего состояния репозитория.
    """

    def __init__(self, client: OpenRouterClient, cfg: AppConfig, model: str | None = None) -> None:
        self._client = client
        self._cfg = cfg
        # Читаем конфиг напрямую, чтобы получить незатронутые __main__.py значения
        from .config import AppConfig as _Cfg
        _loaded = _Cfg.load()
        self._openrouter_model = _loaded.default_model
        self._openrouter_api_key = _loaded.api_key

        # Если есть OpenRouter API-ключ — предпочитаем использовать его для ассистента,
        # чтобы не зависеть от того, запущен ли локальный LLM-сервер
        if self._openrouter_api_key:
            self._preferred_client = OpenRouterClient(
                self._openrouter_api_key, base_url=OPENROUTER_URL
            )
            self._preferred_model = self._openrouter_model
            self._own_client = True
        else:
            # Нет API-ключа — используем переданный клиент (локальный или нет)
            self._preferred_client = client
            self._preferred_model = model or cfg.default_model
            self._own_client = False

    def close(self) -> None:
        if self._own_client:
            self._preferred_client.close()

    @cached_property
    def _index(self):
        from .rag.project_index import ProjectRagIndex
        return ProjectRagIndex(project_root=_PROJECT_ROOT)

    def ask(self, question: str) -> str:
        """Ответить на вопрос о проекте используя RAG + git-контекст."""
        git_info = _get_git_info()
        doc_chunks = self._index.search(question, top_k=5)
        doc_context = "\n\n---\n\n".join(doc_chunks) if doc_chunks else "(документация не найдена)"

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            git_info=git_info,
            doc_context=doc_context,
        )

        from .models import ChatMessage
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=question),
        ]

        return self._send_with_fallback(messages)

    def _send_with_fallback(self, messages) -> str:
        """Отправить запрос через предпочтительный клиент."""
        return self._preferred_client.send(
            messages=messages,
            model=self._preferred_model,
            temperature=0.3,
        )

    def rebuild_index(self) -> None:
        """Пересобрать RAG-индекс по документации."""
        self._index.rebuild()


def _get_git_info() -> str:
    """Получить текущую ветку, последний коммит и список изменённых файлов."""
    parts: list[str] = []

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(_PROJECT_ROOT), "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        parts.append(f"Ветка: {branch or '(detached HEAD)'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        parts.append("Ветка: неизвестна")

    try:
        last_commit = subprocess.check_output(
            ["git", "-C", str(_PROJECT_ROOT), "log", "-1", "--oneline"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        parts.append(f"Последний коммит: {last_commit or 'нет коммитов'}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        parts.append("Последний коммит: неизвестен")

    try:
        status_out = subprocess.check_output(
            ["git", "-C", str(_PROJECT_ROOT), "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        changed = status_out.splitlines() if status_out else []
        if changed:
            files_str = ", ".join(line.split()[-1] for line in changed[:8])
            suffix = f" (и ещё {len(changed) - 8}...)" if len(changed) > 8 else ""
            parts.append(f"Изменённые файлы ({len(changed)}): {files_str}{suffix}")
        else:
            parts.append("Изменённых файлов: нет")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return "\n".join(parts)
