"""Ассистент технической поддержки пользователей.

Объединяет три источника знаний:
1. RAG — поиск по FAQ и документации через FAISS
2. CRM — данные тикетов и пользователей из JSON (через mcp_server_support)
3. Agent + LLM — генерация финального ответа

Точка входа для демо: run_support_demo()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()

_PROJECT_ROOT = Path(__file__).parents[2]
_CRM_DIR = _PROJECT_ROOT / "data" / "crm"
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"

_SUPPORT_SYSTEM_PROMPT = """\
Ты — ассистент технической поддержки. Твоя задача — помочь пользователю решить проблему.

У тебя есть доступ к:
1. Базе знаний (FAQ и документация) — фрагменты приведены ниже в контексте вопроса.
2. Данным тикета — если к вопросу прикреплён тикет, его детали указаны в контексте.

Правила ответа:
- Отвечай конкретно и по делу, опираясь на предоставленный контекст.
- Если тикет содержит специфику (например, конкретный тип ошибки), обязательно учти это.
- Предлагай чёткие шаги для решения проблемы.
- Если информации недостаточно — честно скажи об этом и предложи обратиться в поддержку.
- Язык ответа: русский.\
"""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_ticket_context(ticket_id: str) -> str:
    """Получить контекст тикета из CRM напрямую."""
    tickets_path = _CRM_DIR / "tickets.json"
    users_path = _CRM_DIR / "users.json"

    if not tickets_path.exists():
        return ""

    tickets = _load_json(tickets_path)
    users = _load_json(users_path) if users_path.exists() else []
    user_map = {u["id"]: u for u in users}

    ticket = next((t for t in tickets if t["id"] == ticket_id), None)
    if not ticket:
        return f"(тикет {ticket_id} не найден в CRM)"

    user = user_map.get(ticket["user_id"])
    user_info = (
        f"{user['name']}, email: {user['email']}, тариф: {user['plan'].upper()}, "
        f"компания: {user.get('company') or 'физ. лицо'}"
        if user
        else f"user_id={ticket['user_id']}"
    )

    resolution = ticket.get("resolution", "")
    resolution_line = f"\nРешение: {resolution}" if resolution else ""
    tags = ", ".join(ticket.get("tags", []))

    return (
        f"=== Данные тикета {ticket['id']} ===\n"
        f"Тема: {ticket['subject']}\n"
        f"Тип проблемы: {ticket['type']}\n"
        f"Статус: {ticket['status']} | Приоритет: {ticket['priority']}\n"
        f"Пользователь: {user_info}\n"
        f"Создан: {ticket['created_at']}\n"
        f"Теги: {tags}\n"
        f"\nОписание пользователя:\n{ticket['description']}"
        f"{resolution_line}\n"
        f"{'=' * 40}"
    )


def _list_tickets_table() -> str:
    """Вернуть форматированный список всех тикетов."""
    tickets_path = _CRM_DIR / "tickets.json"
    users_path = _CRM_DIR / "users.json"

    tickets = _load_json(tickets_path)
    users = _load_json(users_path)
    user_map = {u["id"]: u for u in users}

    return tickets, user_map


class SupportAssistant:
    """Ассистент поддержки: RAG + CRM + LLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        index_strategy: str = "structural",
        top_k: int = 4,
        temperature: float = 0.3,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._index_strategy = index_strategy
        self._top_k = top_k
        self._temperature = temperature
        self._base_url = base_url

        self._rag_agent: Any = None

    def _ensure_rag(self) -> None:
        if self._rag_agent is not None:
            return
        from .rag.rag_agent import RagAgent
        from .models import OPENROUTER_URL

        index_dir = _INDEX_DIR
        self._rag_agent = RagAgent(
            api_key=self._api_key,
            model=self._model,
            index_dir=index_dir,
            strategy=self._index_strategy,
            top_k=self._top_k,
            temperature=self._temperature,
            base_url=self._base_url or OPENROUTER_URL,
        )

    def _build_prompt(
        self,
        question: str,
        faq_chunks: list[tuple[Any, float]],
        ticket_context: str,
    ) -> str:
        lines: list[str] = []

        if ticket_context:
            lines.append(ticket_context)
            lines.append("")

        if faq_chunks:
            lines.append("=== Релевантные фрагменты из FAQ и документации ===")
            for i, (chunk, score) in enumerate(faq_chunks, 1):
                section = chunk.metadata.get("section", "")
                title = chunk.metadata.get("title", "")
                label_parts = [p for p in [title, section] if p]
                label = " · ".join(label_parts) if label_parts else "FAQ"
                lines.append(f"\n[{i}] {label} (релевантность={score:.2f})")
                lines.append(chunk.text.strip())
            lines.append("\n" + "=" * 40)
            lines.append("")

        lines.append(f"Вопрос пользователя: {question}")
        lines.append("")
        lines.append(
            "Дай подробный, конкретный ответ на русском языке. "
            "Если тикет содержит специфику — обязательно учти её. "
            "Предложи пошаговое решение. "
            "Верни ТОЛЬКО ответ без дополнительных секций."
        )
        return "\n".join(lines)

    def answer(self, question: str, ticket_id: str | None = None) -> dict[str, Any]:
        """Ответить на вопрос с учётом контекста тикета (если указан).

        Returns:
            dict с ключами: answer, ticket_context, faq_chunks_count, model
        """
        self._ensure_rag()

        ticket_context = ""
        if ticket_id:
            ticket_context = _get_ticket_context(ticket_id)

        from .rag.rag_agent import RagAgent
        assert isinstance(self._rag_agent, RagAgent)

        rag_answer = self._rag_agent.ask(question, use_rag=True)
        faq_chunks = rag_answer.chunks

        from .agent import Agent
        from .api import OpenRouterClient
        from .models import OPENROUTER_URL

        prompt = self._build_prompt(question, faq_chunks, ticket_context)

        with OpenRouterClient(
            self._api_key, base_url=self._base_url or OPENROUTER_URL
        ) as client:
            agent = Agent(
                client=client,
                model=self._model,
                temperature=self._temperature,
                system_prompt=_SUPPORT_SYSTEM_PROMPT,
                compression_enabled=False,
            )
            agent._raw_history = [m for m in agent._raw_history if m.role == "system"]
            answer_text = agent.run(prompt).strip()

        return {
            "answer": answer_text,
            "ticket_context": ticket_context,
            "faq_chunks_count": len(faq_chunks),
            "model": self._model,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Демо
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_SCENARIOS = [
    {
        "title": "Проблема с авторизацией",
        "ticket_id": "TKT-001",
        "question": "Почему не работает авторизация? Пароль правильный, но всё равно выдаёт ошибку.",
    },
    {
        "title": "Проблемы с оплатой подписки",
        "ticket_id": "TKT-002",
        "question": "Как перейти на платный план? Оплата не проходит.",
    },
    {
        "title": "Ошибка 401 при вызове API",
        "ticket_id": "TKT-005",
        "question": "Почему API возвращает 401 Unauthorized при использовании нового ключа?",
    },
    {
        "title": "Превышение rate limit",
        "ticket_id": "TKT-008",
        "question": "Получаем 429 Too Many Requests, хотя в нашем плане должен быть высокий лимит.",
    },
]


def _print_tickets_overview() -> None:
    """Вывести обзор всех тикетов из CRM."""
    tickets, user_map = _list_tickets_table()

    table = Table(
        title="CRM — Активные тикеты поддержки",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="bold", width=10)
    table.add_column("Приоритет", width=10)
    table.add_column("Тип", width=14)
    table.add_column("Статус", width=12)
    table.add_column("Пользователь", width=20)
    table.add_column("Тема", width=40)

    priority_styles = {"high": "red", "medium": "yellow", "low": "green"}
    status_labels = {"open": "открыт", "in_progress": "в работе", "resolved": "решён"}

    for t in tickets:
        user = user_map.get(t["user_id"])
        user_name = user["name"] if user else t["user_id"]
        plan_badge = f" [{user['plan'].upper()}]" if user else ""
        priority_style = priority_styles.get(t["priority"], "white")

        table.add_row(
            t["id"],
            f"[{priority_style}]{t['priority']}[/{priority_style}]",
            t["type"],
            status_labels.get(t["status"], t["status"]),
            f"{user_name}{plan_badge}",
            t["subject"],
        )

    console.print()
    console.print(table)
    console.print()


def _print_answer(scenario: dict, result: dict, idx: int, total: int) -> None:
    """Красиво вывести ответ ассистента."""
    ticket_id = scenario.get("ticket_id", "")
    question = scenario["question"]

    console.print(
        Rule(
            f"[bold cyan]Сценарий {idx}/{total}: {scenario['title']}[/bold cyan]",
            style="cyan",
        )
    )

    console.print(
        Panel(
            f"[bold yellow]Вопрос:[/bold yellow] {question}\n"
            f"[dim]Тикет: {ticket_id} | FAQ-фрагменты: {result['faq_chunks_count']} | Модель: {result['model']}[/dim]",
            border_style="yellow",
            expand=True,
        )
    )

    if result["ticket_context"]:
        console.print(
            Panel(
                result["ticket_context"],
                title="[bold blue]Контекст тикета из CRM[/bold blue]",
                border_style="blue",
                expand=True,
            )
        )

    console.print(
        Panel(
            result["answer"],
            title="[bold green]Ответ ассистента поддержки[/bold green]",
            border_style="green",
            expand=True,
        )
    )
    console.print()


def run_support_demo(api_key: str, model: str, base_url: str | None = None) -> None:
    """Запустить полное демо ассистента поддержки для записи видео."""
    console.print(
        Panel(
            "[bold cyan]AI-Ассистент поддержки пользователей[/bold cyan]\n"
            "[dim]RAG (FAQ + документация) + CRM (тикеты) + LLM[/dim]",
            border_style="cyan",
            expand=False,
        )
    )

    console.print(Rule("[bold]Шаг 1: Загружаем данные из CRM[/bold]", style="white"))
    _print_tickets_overview()
    time.sleep(1)

    console.print(Rule("[bold]Шаг 2: Инициализируем ассистента[/bold]", style="white"))
    console.print("[dim]Загружаем RAG-индекс (FAQ + документация)...[/dim]")

    assistant = SupportAssistant(
        api_key=api_key,
        model=model,
        base_url=base_url,
        top_k=4,
        temperature=0.3,
    )
    assistant._ensure_rag()
    console.print("[green]✓[/green] RAG-индекс загружен\n")
    time.sleep(0.5)

    console.print(Rule("[bold]Шаг 3: Демо-сценарии[/bold]", style="white"))
    console.print("[dim]Ассистент будет отвечать на реальные тикеты из CRM[/dim]\n")
    time.sleep(1)

    total = len(_DEMO_SCENARIOS)
    for idx, scenario in enumerate(_DEMO_SCENARIOS, 1):
        console.print(
            f"[dim]⏳ Обрабатываем тикет {scenario.get('ticket_id', '')}...[/dim]"
        )
        result = assistant.answer(
            question=scenario["question"],
            ticket_id=scenario.get("ticket_id"),
        )
        _print_answer(scenario, result, idx, total)
        if idx < total:
            time.sleep(1)

    console.print(Rule("[bold]Шаг 4: Обновление статуса тикета[/bold]", style="white"))
    _demo_update_ticket_status("TKT-001", "in_progress")

    console.print(
        Panel(
            "[bold green]✓ Демо завершено![/bold green]\n\n"
            "Ассистент поддержки успешно продемонстрировал:\n"
            "  • [cyan]RAG[/cyan] — поиск релевантных ответов в FAQ и документации\n"
            "  • [blue]CRM[/blue] — учёт контекста тикета (тип проблемы, тариф, описание)\n"
            "  • [green]LLM[/green] — генерация конкретных пошаговых ответов\n"
            "  • [yellow]MCP[/yellow] — обновление статуса тикетов в реальном времени",
            border_style="green",
            expand=False,
        )
    )


def _demo_update_ticket_status(ticket_id: str, new_status: str) -> None:
    """Показать обновление статуса тикета."""
    from .mcp_server_support import update_ticket_status

    console.print(
        f"[dim]Обновляем статус тикета [bold]{ticket_id}[/bold] → [bold]{new_status}[/bold]...[/dim]"
    )
    result = update_ticket_status(ticket_id, new_status)
    console.print(
        Panel(
            result,
            title="[bold yellow]MCP: update_ticket_status[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    console.print()
