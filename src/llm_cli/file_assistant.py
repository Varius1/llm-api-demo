"""Ассистент для работы с файлами проекта через MCP + LLM tool calling.

Ассистент сам инициирует работу с файлами, получая задачу на уровне цели.
Поддерживаемые сценарии:
  - Поиск использования компонента/API по всему проекту
  - Генерация/обновление документации (README, CHANGELOG, ADR)
  - Проверка соответствия файлов правилам и инвариантам
  - Создание diff и списка изменений
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .api import OpenRouterClient
from .models import ChatMessage, LOCAL_BASE_URL, LOCAL_MODEL_ID, OPENROUTER_URL

console = Console()

_PROJECT_ROOT = Path(__file__).parents[2].resolve()

_SYSTEM_PROMPT = """\
Ты — ассистент для работы с файлами проекта llm-api-demo.
У тебя есть набор MCP-инструментов для работы с файловой системой:

  • read_file(path)                            — прочитать содержимое файла
  • list_files(directory, pattern)             — найти файлы по glob-паттерну
  • grep_in_files(pattern, directory, file_glob, context_lines, max_matches)
                                               — поиск строки/regex по файлам
  • write_file(path, content, create_dirs)     — создать/обновить файл
  • get_file_diff(path, new_content)           — показать diff перед записью
  • get_project_structure(directory, depth)    — дерево каталогов
  • get_git_log(n, oneline)                    — история git-коммитов

Правила работы:
1. ВСЕГДА начинай с изучения структуры проекта — вызови get_project_structure или list_files.
2. Для анализа кода используй grep_in_files — ищи по нескольким файлам сразу.
3. Перед записью файла покажи diff через get_file_diff, чтобы пользователь видел изменения.
4. Работай автономно: сам выбирай нужные файлы, не проси пользователя указывать пути.
5. Когда задача выполнена — дай краткий итог: что сделано, какие файлы затронуты.
6. Отвечай на русском языке.
"""

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_MAX_TOOL_ROUNDS = 15  # защита от бесконечного цикла


class FileAssistant:
    """Ассистент, который активно работает с файлами проекта через MCP-инструменты.

    Использование:
        async with FileAssistant(api_key="...", base_url="...", model="...") as fa:
            result = await fa.run("найди все места где используется MCPSession")
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = OPENROUTER_URL,
        verbose: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._verbose = verbose
        self._mcp: Any = None
        self._client: OpenRouterClient | None = None

    async def __aenter__(self) -> FileAssistant:
        from .mcp_client import MCPFilesSession

        self._mcp = MCPFilesSession()
        await self._mcp.__aenter__()
        self._client = OpenRouterClient(self._api_key, base_url=self._base_url)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(*args)
        if self._client is not None:
            self._client.close()

    async def run(self, goal: str) -> str:
        """Выполнить задачу с автономным использованием файловых инструментов.

        goal — задача на уровне цели (например: "найди все импорты httpx")
        Возвращает финальный ответ ассистента.
        """
        assert self._mcp is not None, "Используйте как async context manager"
        assert self._client is not None

        tools = self._mcp.get_tools_schema()

        if self._verbose:
            _print_tools_banner(tools)
            console.print(Rule("[bold yellow]Tool Calling Loop[/bold yellow]", style="yellow"))
            console.print()

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=goal),
        ]

        for round_n in range(_MAX_TOOL_ROUNDS):
            if self._verbose:
                console.print(
                    f"  [dim cyan]→ LLM запрос #{round_n + 1} "
                    f"(инструментов: {len(tools)})[/dim cyan]"
                )

            try:
                chat_response = self._client.send_raw(
                    messages=messages,
                    model=self._model,
                    temperature=0.2,
                    tools=tools if tools else None,
                )
            except RuntimeError as e:
                err_str = str(e)
                if "401" in err_str or "Authentication" in err_str:
                    raise RuntimeError(
                        "Ошибка аутентификации OpenRouter (401).\n"
                        "Проверьте API-ключ: https://openrouter.ai/keys\n"
                        "Установите: export OPENROUTER_API_KEY=sk-or-...\n"
                        f"Детали: {err_str}"
                    ) from e
                raise

            if chat_response.error is not None:
                raise RuntimeError(f"API ошибка: {chat_response.error.message}")
            if not chat_response.choices:
                raise RuntimeError("Нет ответа в choices")

            choice = chat_response.choices[0]
            finish_reason = choice.finish_reason
            msg = choice.message

            # Нет tool calls — финальный ответ
            if finish_reason != "tool_calls" or not msg.tool_calls:
                content = msg.content or ""
                if self._verbose:
                    console.print("  [dim green]✓ Финальный ответ получен[/dim green]")
                    console.print()
                    console.print(Rule("[bold green]Итог[/bold green]", style="green"))
                    console.print(Panel(content, border_style="green", expand=False))
                return content

            # Показываем вызовы инструментов
            if self._verbose:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    _print_tool_call(tc.function.name, args)

            # Добавляем assistant-сообщение с tool_calls
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                )
            )

            # Вызываем каждый инструмент через MCP
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                tool_result = await self._mcp.call_tool(tc.function.name, args)

                if self._verbose:
                    _print_tool_result(tc.function.name, tool_result)

                messages.append(
                    ChatMessage(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )

        fallback = "Достигнут лимит итераций tool calling. Ассистент не завершил задачу."
        if self._verbose:
            console.print(f"[yellow]{fallback}[/yellow]")
        return fallback


def _print_tools_banner(tools: list[Any]) -> None:
    from rich.table import Table

    table = Table(
        title=f"Файловые инструменты ({len(tools)})",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Инструмент", style="bold yellow", no_wrap=True)
    table.add_column("Описание", style="white")
    for t in tools:
        desc = t.function.description.split("\n")[0]
        table.add_row(t.function.name, desc)
    console.print()
    console.print(table)
    console.print()


def _print_tool_call(name: str, arguments: dict[str, Any]) -> None:
    args_parts = []
    for k, v in arguments.items():
        val_str = str(v)
        val_repr = repr(val_str[:60] + "..." if len(val_str) > 60 else val_str)
        args_parts.append(f"{k}={val_repr}")
    args_str = ", ".join(args_parts)
    console.print(
        Text.assemble(
            ("  ▶ ", "bold green"),
            (f"{name}(", "bold white"),
            (args_str, "dim white"),
            (")", "bold white"),
        )
    )


def _print_tool_result(name: str, result: str) -> None:
    lines = result.splitlines()
    preview = lines[:6]
    suffix = f"\n  [dim]... ещё {len(lines) - 6} строк[/dim]" if len(lines) > 6 else ""
    preview_text = "\n".join(f"  [cyan]◀[/cyan] [italic]{line}[/italic]" for line in preview)
    console.print(preview_text + suffix)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Интерактивный режим
# ─────────────────────────────────────────────────────────────────────────────

_WELCOME_BANNER = """\
[bold cyan]File Assistant[/bold cyan] — ассистент для работы с файлами проекта

Задавайте цели, ассистент сам найдёт нужные файлы и выполнит задачу.

[bold]Примеры задач:[/bold]
  • найди все места где используется класс Agent
  • сгенерируй CHANGELOG.md на основе git-истории
  • проверь, нет ли прямых импортов httpx вне api.py
  • покажи структуру проекта и список Python-модулей
  • обнови README: добавь секцию про file-assistant команду

[dim]Команды:[/dim] [bold]/exit[/bold] или [bold]/quit[/bold] — выход, [bold]/help[/bold] — эта справка
"""


async def _run_interactive(api_key: str, model: str, base_url: str) -> None:
    mode_label = "[dim]локальная модель[/dim]" if base_url != OPENROUTER_URL else "[dim]OpenRouter[/dim]"
    console.print(Panel(
        _WELCOME_BANNER + f"\nРежим: {mode_label}  |  Модель: [dim]{model}[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    async with FileAssistant(api_key=api_key, model=model, base_url=base_url) as fa:
        while True:
            try:
                console.print("[bold cyan]Задача>[/bold cyan] ", end="")
                goal = input().strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Выход.[/dim]")
                break

            if not goal:
                continue
            if goal.lower() in {"/exit", "/quit", "exit", "quit"}:
                console.print("[dim]До свидания![/dim]")
                break
            if goal.lower() in {"/help", "help", "?"}:
                console.print(Panel(_WELCOME_BANNER, border_style="cyan", expand=False))
                continue

            console.print()
            console.print(
                Panel(
                    f"[bold white]{goal}[/bold white]",
                    title="[bold yellow]Задача[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                )
            )
            console.print()

            try:
                await fa.run(goal)
            except Exception as e:
                console.print(f"[red]Ошибка выполнения задачи: {e}[/red]")

            console.print()


def run_file_assistant(api_key: str, model: str = _DEFAULT_MODEL, base_url: str = OPENROUTER_URL) -> None:
    """Запустить интерактивный File Assistant."""
    asyncio.run(_run_interactive(api_key, model, base_url))


# ─────────────────────────────────────────────────────────────────────────────
# Неинтерактивный режим (для скриптов / демо)
# ─────────────────────────────────────────────────────────────────────────────

async def _run_goal(api_key: str, model: str, base_url: str, goal: str) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold white]{goal}[/bold white]",
            title="[bold yellow]Задача[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    console.print()
    async with FileAssistant(api_key=api_key, model=model, base_url=base_url) as fa:
        await fa.run(goal)


def run_file_assistant_goal(
    api_key: str,
    goal: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = OPENROUTER_URL,
) -> None:
    """Выполнить одну задачу неинтерактивно (для скриптов и демо)."""
    asyncio.run(_run_goal(api_key, model, base_url, goal))
