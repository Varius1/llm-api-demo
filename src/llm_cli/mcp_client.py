"""MCP-клиент: подключается к локальному серверу и получает список инструментов."""

from __future__ import annotations

import asyncio
import json
import sys
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from .models import ToolDefinition, ToolDefinitionFunction

console = Console()


class MCPSession:
    """Постоянная MCP-сессия для использования в чате.

    Использование:
        async with MCPSession() as session:
            tools = session.get_tools_schema()
            result = await session.call_tool("get_weather", {"city": "Москва"})
    """

    def __init__(self, python_executable: str | None = None) -> None:
        self._python = python_executable or sys.executable
        self._session: ClientSession | None = None
        self._tools_schema: list[ToolDefinition] | None = None
        self._exit_stack: Any = None
        self._loop: Any = None

    async def __aenter__(self) -> MCPSession:
        import asyncio as _asyncio
        from contextlib import AsyncExitStack

        self._loop = _asyncio.get_running_loop()

        server_params = StdioServerParameters(
            command=self._python,
            args=["-m", "llm_cli.mcp_server"],
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        await self._refresh_tools()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()

    async def _refresh_tools(self) -> None:
        assert self._session is not None
        result = await self._session.list_tools()
        self._tools_schema = [
            ToolDefinition(
                function=ToolDefinitionFunction(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema or {"type": "object", "properties": {}},
                )
            )
            for tool in result.tools
        ]

    def get_tools_schema(self) -> list[ToolDefinition]:
        """Вернуть список инструментов в OpenAI-формате."""
        return list(self._tools_schema or [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Вызвать инструмент и вернуть результат как строку."""
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        content = result.content
        if not content:
            return ""
        first = content[0]
        if hasattr(first, "text"):
            return first.text
        return str(first)

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> str:
        """Синхронная версия вызова инструмента (запускает asyncio.run)."""
        assert self._loop is not None
        result = asyncio.run_coroutine_threadsafe(
            self.call_tool(name, arguments), self._loop
        )
        return result.result()


_DEMO_CALLS: list[tuple[str, dict[str, str]]] = [
    ("get_weather", {"city": "Москва"}),
    ("calculate", {"expression": "(100 + 200) * 3 / 5"}),
    ("list_models", {}),
]


async def _run(python_executable: str) -> None:
    server_params = StdioServerParameters(
        command=python_executable,
        args=["-m", "llm_cli.mcp_server"],
    )

    console.print(
        Panel(
            "[bold cyan]Устанавливаю MCP-соединение...[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            console.print("[green]✓[/green] Соединение установлено\n")

            # ── Шаг 1: список инструментов ──────────────────────────────────
            result = await session.list_tools()
            tools = result.tools

            table = Table(
                title=f"Доступные MCP-инструменты ({len(tools)})",
                box=box.ROUNDED,
                border_style="cyan",
                header_style="bold magenta",
                show_lines=True,
            )
            table.add_column("Инструмент", style="bold yellow", no_wrap=True)
            table.add_column("Описание", style="white")
            table.add_column("Параметры", style="dim")

            for tool in tools:
                params_schema = tool.inputSchema or {}
                props = params_schema.get("properties", {})
                required = params_schema.get("required", [])

                if props:
                    param_parts = []
                    for name, info in props.items():
                        req = " *" if name in required else ""
                        type_str = info.get("type", "any")
                        param_parts.append(f"{name}: {type_str}{req}")
                    params_str = "\n".join(param_parts)
                else:
                    params_str = "—"

                table.add_row(tool.name, tool.description or "", params_str)

            console.print(table)
            console.print("[dim]* — обязательный параметр[/dim]\n")

            # ── Шаг 2: вызов инструментов ───────────────────────────────────
            console.print(
                Rule("[bold magenta]Вызов инструментов[/bold magenta]", style="magenta")
            )
            console.print()

            for tool_name, arguments in _DEMO_CALLS:
                args_str = ", ".join(f'{k}="{v}"' for k, v in arguments.items())
                console.print(
                    Text.assemble(
                        ("  ▶ ", "bold green"),
                        (f"{tool_name}({args_str})", "bold white"),
                    )
                )

                call_result = await session.call_tool(tool_name, arguments)
                content = call_result.content
                if content:
                    raw = (
                        content[0].text
                        if hasattr(content[0], "text")
                        else str(content[0])
                    )
                else:
                    raw = "(пустой ответ)"

                for line in raw.splitlines():
                    console.print(f"  [cyan]◀[/cyan] [italic]{line}[/italic]")
                console.print()


def run_mcp_demo() -> None:
    python_exec = sys.executable
    asyncio.run(_run(python_exec))


_AGENT_DEMO_PROMPT = (
    "Используй инструмент get_crypto_price и узнай текущий курс следующих криптовалют в USD:\n"
    "- BTC (биткоин)\n"
    "- ETH (эфириум)\n"
    "- SOL (солана)\n"
    "Дай краткий итог с актуальными ценами."
)

_AGENT_DEMO_MODEL = "openai/gpt-4o-mini"


async def _run_agent_demo(api_key: str) -> None:
    from .agent import Agent
    from .api import OpenRouterClient

    console.print(
        Panel(
            "[bold cyan]Запускаю агента с MCP-инструментами...[/bold cyan]\n"
            "[dim]Агент автоматически вызовет инструменты и вернёт результат[/dim]",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    async with MCPSession() as mcp:
        tools = mcp.get_tools_schema()
        tool_names = [t.function.name for t in tools]

        table = Table(
            title=f"MCP-инструменты ({len(tools)})",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Инструмент", style="bold yellow", no_wrap=True)
        table.add_column("Описание", style="white")
        for t in tools:
            table.add_row(t.function.name, t.function.description)
        console.print(table)
        console.print(f"[dim]Модель: {_AGENT_DEMO_MODEL}[/dim]\n")

        console.print(
            Rule("[bold magenta]Промпт агенту[/bold magenta]", style="magenta")
        )
        console.print(
            Panel(
                _AGENT_DEMO_PROMPT,
                border_style="dim",
                expand=False,
            )
        )
        console.print()
        console.print(
            Rule("[bold yellow]Tool Calling Loop[/bold yellow]", style="yellow")
        )
        console.print()

        with OpenRouterClient(api_key) as client:
            agent = Agent(client=client, model=_AGENT_DEMO_MODEL, mcp_session=mcp)
            reply = await agent.run_async(_AGENT_DEMO_PROMPT)

        console.print()
        console.print(
            Rule("[bold green]Финальный ответ LLM[/bold green]", style="green")
        )
        console.print(Panel(reply, border_style="green", expand=False))


def run_agent_demo() -> None:
    """Автоматический сценарий: агент вызывает MCP-инструменты и возвращает результат."""
    from .config import ensure_config

    cfg = ensure_config()
    asyncio.run(_run_agent_demo(cfg.api_key))


_PIPELINE_DEMO_PROMPT = (
    "Выполни следующий пайплайн по шагам:\n"
    "1. Используй инструмент search чтобы найти информацию о Python (запрос: 'python', max_results=5).\n"
    "2. Используй инструмент summarize чтобы сжать полученный текст до 3 предложений.\n"
    "3. Используй инструмент save_to_file чтобы сохранить резюме в файл pipeline_result.txt.\n"
    "После выполнения всех шагов дай краткий итог: что нашёл, что получилось в резюме, куда сохранил."
)

_PIPELINE_DEMO_MODEL = "openai/gpt-4o-mini"


async def _run_pipeline_demo(api_key: str) -> None:
    from .agent import Agent
    from .api import OpenRouterClient

    console.print(
        Panel(
            "[bold cyan]MCP Pipeline Demo[/bold cyan]\n"
            "[dim]search → summarize → save_to_file[/dim]\n\n"
            "[white]Агент автоматически выполнит цепочку инструментов:[/white]\n"
            "  [yellow]1.[/yellow] [bold]search[/bold]       — получить данные\n"
            "  [yellow]2.[/yellow] [bold]summarize[/bold]    — обработать / сжать\n"
            "  [yellow]3.[/yellow] [bold]save_to_file[/bold] — сохранить результат",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    async with MCPSession() as mcp:
        tools = mcp.get_tools_schema()

        pipeline_tools = ["search", "summarize", "save_to_file"]
        pipeline_tool_objs = [t for t in tools if t.function.name in pipeline_tools]
        other_tools_count = len(tools) - len(pipeline_tool_objs)

        table = Table(
            title=f"Pipeline-инструменты ({len(pipeline_tool_objs)} из {len(tools)} доступных)",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("#", style="bold yellow", width=3)
        table.add_column("Инструмент", style="bold green", no_wrap=True)
        table.add_column("Описание", style="white")

        for i, t in enumerate(pipeline_tool_objs, 1):
            desc_short = t.function.description.split("\n")[0]
            table.add_row(str(i), t.function.name, desc_short)

        console.print(table)
        console.print(
            f"[dim]+ ещё {other_tools_count} инструментов зарегистрировано на сервере[/dim]\n"
        )

        console.print(
            Rule("[bold magenta]Промпт агенту[/bold magenta]", style="magenta")
        )
        console.print(
            Panel(
                _PIPELINE_DEMO_PROMPT,
                border_style="dim",
                expand=False,
            )
        )
        console.print()
        console.print(
            Rule("[bold yellow]Tool Calling Pipeline[/bold yellow]", style="yellow")
        )
        console.print()

        with OpenRouterClient(api_key) as client:
            agent = Agent(client=client, model=_PIPELINE_DEMO_MODEL, mcp_session=mcp)
            reply = await agent.run_async(_PIPELINE_DEMO_PROMPT)

        console.print()
        console.print(
            Rule("[bold green]Финальный ответ LLM[/bold green]", style="green")
        )
        console.print(Panel(reply, border_style="green", expand=False))

        import os
        from pathlib import Path

        result_file = Path(os.getcwd()) / "pipeline_result.txt"
        if result_file.exists():
            content = result_file.read_text(encoding="utf-8")
            console.print()
            console.print(
                Rule(
                    "[bold blue]Содержимое сохранённого файла[/bold blue]", style="blue"
                )
            )
            console.print(
                Panel(
                    content,
                    title=f"[dim]{result_file}[/dim]",
                    border_style="blue",
                    expand=False,
                )
            )


def run_pipeline_demo() -> None:
    """Автоматический пайплайн: search → summarize → save_to_file."""
    from .config import ensure_config

    cfg = ensure_config()
    asyncio.run(_run_pipeline_demo(cfg.api_key))


# ─────────────────────────────────────────────────────────────────────────────
# MultiMCPSession — оркестратор нескольких MCP-серверов
# ─────────────────────────────────────────────────────────────────────────────


class MultiMCPSession(MCPSession):
    """Оркестратор двух MCP-серверов с автоматической маршрутизацией вызовов.

    Сервер 1 — Data & Analytics (data-analytics):
        search, get_crypto_price, get_weather, calculate, summarize

    Сервер 2 — Tools & Storage (tools-storage):
        save_to_file, list_models, add_reminder, get_pending_reminders,
        start_price_monitor, get_price_summary

    Использование:
        async with MultiMCPSession() as session:
            tools = session.get_tools_schema()
            result = await session.call_tool("get_weather", {"city": "Москва"})
    """

    _DATA_SERVER_MODULE = "llm_cli.mcp_server_data"
    _TOOLS_SERVER_MODULE = "llm_cli.mcp_server_tools"

    def __init__(self, python_executable: str | None = None) -> None:
        self._python = python_executable or sys.executable
        self._data_session: ClientSession | None = None
        self._tools_session: ClientSession | None = None
        self._tools_schema: list[ToolDefinition] = []
        self._tool_to_server: dict[str, str] = {}  # tool_name -> "data" | "tools"
        self._exit_stack: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> MultiMCPSession:
        from contextlib import AsyncExitStack

        self._loop = asyncio.get_running_loop()
        self._exit_stack = AsyncExitStack()

        # Запускаем Data & Analytics сервер
        data_params = StdioServerParameters(
            command=self._python,
            args=["-m", self._DATA_SERVER_MODULE],
        )
        data_read, data_write = await self._exit_stack.enter_async_context(
            stdio_client(data_params)
        )
        self._data_session = await self._exit_stack.enter_async_context(
            ClientSession(data_read, data_write)
        )
        await self._data_session.initialize()

        # Запускаем Tools & Storage сервер
        tools_params = StdioServerParameters(
            command=self._python,
            args=["-m", self._TOOLS_SERVER_MODULE],
        )
        tools_read, tools_write = await self._exit_stack.enter_async_context(
            stdio_client(tools_params)
        )
        self._tools_session = await self._exit_stack.enter_async_context(
            ClientSession(tools_read, tools_write)
        )
        await self._tools_session.initialize()

        await self._refresh_tools()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()

    async def _refresh_tools(self) -> None:
        assert self._data_session is not None
        assert self._tools_session is not None

        self._tools_schema = []
        self._tool_to_server = {}

        # Инструменты Data-сервера
        data_result = await self._data_session.list_tools()
        for tool in data_result.tools:
            self._tool_to_server[tool.name] = "data"
            self._tools_schema.append(
                ToolDefinition(
                    function=ToolDefinitionFunction(
                        name=tool.name,
                        description=f"[Data] {tool.description or ''}",
                        parameters=tool.inputSchema
                        or {"type": "object", "properties": {}},
                    )
                )
            )

        # Инструменты Tools-сервера
        tools_result = await self._tools_session.list_tools()
        for tool in tools_result.tools:
            self._tool_to_server[tool.name] = "tools"
            self._tools_schema.append(
                ToolDefinition(
                    function=ToolDefinitionFunction(
                        name=tool.name,
                        description=f"[Tools] {tool.description or ''}",
                        parameters=tool.inputSchema
                        or {"type": "object", "properties": {}},
                    )
                )
            )

    def get_tools_schema(self) -> list[ToolDefinition]:
        """Вернуть объединённый список инструментов обоих серверов."""
        return list(self._tools_schema)

    def get_tool_server(self, name: str) -> str | None:
        """Вернуть имя сервера для инструмента: 'data' или 'tools'."""
        return self._tool_to_server.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Вызвать инструмент, автоматически маршрутизируя на нужный сервер."""
        server_key = self._tool_to_server.get(name)
        if server_key == "data":
            session = self._data_session
        elif server_key == "tools":
            session = self._tools_session
        else:
            return f"Ошибка: инструмент '{name}' не зарегистрирован ни на одном сервере"

        assert session is not None
        result = await session.call_tool(name, arguments)
        content = result.content
        if not content:
            return ""
        first = content[0]
        if hasattr(first, "text"):
            return first.text
        return str(first)

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> str:
        """Синхронная обёртка для вызова из синхронного tool loop (вне event loop)."""
        import concurrent.futures

        if self._loop is not None and self._loop.is_running():
            # Планируем корутину в существующем loop из другого треда
            fut = asyncio.run_coroutine_threadsafe(
                self.call_tool(name, arguments), self._loop
            )
            return fut.result(timeout=60)
        return asyncio.run(self.call_tool(name, arguments))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration Demo — длинный флоу через оба сервера
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATION_DEMO_PROMPT = """\
Ты — умный агент-аналитик. Выполни следующий исследовательский пайплайн строго по шагам, \
используя доступные инструменты с двух серверов:

ШАГИ (выполнять последовательно):

1. Используй инструмент search: найди информацию про MCP (Model Context Protocol), max_results=4.
2. Используй инструмент get_crypto_price: узнай текущую цену BTC в USD.
3. Используй инструмент get_crypto_price: узнай текущую цену ETH в USD.
4. Используй инструмент get_weather: узнай погоду в Москве.
5. Используй инструмент calculate: посчитай выражение "365 * 24 * 60" (минуты в году).
6. Используй инструмент summarize: сожми результаты шага 1 (search) до 2 предложений.
7. Используй инструмент list_models: получи список доступных LLM-моделей.
8. Используй инструмент add_reminder: установи напоминание \
"Проверить курс BTC и ETH" через 120 секунд.
9. Используй инструмент save_to_file: сохрани итоговый отчёт в файл \
orchestration_report.md. Отчёт должен содержать: краткое резюме MCP из шага 6, \
текущие цены BTC и ETH, погоду в Москве, результат вычисления и список моделей.
10. Используй инструмент get_pending_reminders: проверь список всех напоминаний.

После выполнения всех шагов дай финальный итог: что сделал, что нашёл, что сохранил.
"""

_ORCHESTRATION_DEMO_MODEL = "openai/gpt-4o-mini"


async def _run_orchestration_demo(api_key: str) -> None:
    from .agent import Agent
    from .api import OpenRouterClient

    # ── Шапка ───────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            "[bold cyan]MCP Orchestration Demo[/bold cyan]\n"
            "[dim]Два MCP-сервера · Автоматическая маршрутизация · Длинный флоу[/dim]\n\n"
            "[white]Регистрируем два специализированных MCP-сервера:[/white]\n"
            "  [bold yellow]①[/bold yellow] [bold green]Data & Analytics[/bold green]  "
            "— search, get_crypto_price, get_weather, calculate, summarize\n"
            "  [bold yellow]②[/bold yellow] [bold blue]Tools & Storage[/bold blue]    "
            "— save_to_file, list_models, add_reminder, get_pending_reminders, …",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    async with MultiMCPSession() as mcp:
        tools = mcp.get_tools_schema()
        data_tools = [t for t in tools if t.function.description.startswith("[Data]")]
        tools_tools = [t for t in tools if t.function.description.startswith("[Tools]")]

        # ── Таблица инструментов по серверам ────────────────────────────────
        servers_table = Table(
            title=f"Зарегистрированные инструменты ({len(tools)} всего)",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold magenta",
            show_lines=True,
        )
        servers_table.add_column("Сервер", style="bold", no_wrap=True, width=22)
        servers_table.add_column("Инструмент", style="bold yellow", no_wrap=True)
        servers_table.add_column("Описание", style="white")

        for t in data_tools:
            desc = t.function.description.removeprefix("[Data] ").split("\n")[0]
            servers_table.add_row(
                "[green]Data & Analytics[/green]", t.function.name, desc
            )
        for t in tools_tools:
            desc = t.function.description.removeprefix("[Tools] ").split("\n")[0]
            servers_table.add_row("[blue]Tools & Storage[/blue]", t.function.name, desc)

        console.print(servers_table)
        console.print(f"[dim]Модель агента: {_ORCHESTRATION_DEMO_MODEL}[/dim]\n")

        # ── Промпт ──────────────────────────────────────────────────────────
        console.print(
            Rule(
                "[bold magenta]Промпт агенту (10 шагов через 2 сервера)[/bold magenta]",
                style="magenta",
            )
        )
        console.print(
            Panel(
                _ORCHESTRATION_DEMO_PROMPT,
                border_style="dim",
                expand=False,
            )
        )
        console.print()
        console.print(
            Rule(
                "[bold yellow]Orchestration Tool Calling Loop[/bold yellow]",
                style="yellow",
            )
        )
        console.print()

        # Патчим call_tool_sync чтобы выводить маршрут
        original_call_tool = mcp.call_tool

        async def instrumented_call_tool(name: str, arguments: dict[str, Any]) -> str:
            server_key = mcp.get_tool_server(name)
            if server_key == "data":
                server_label = "[green]Data & Analytics[/green]"
            elif server_key == "tools":
                server_label = "[blue]Tools & Storage[/blue]"
            else:
                server_label = "[red]Unknown[/red]"
            console.print(f"  [dim]  ↳ Маршрут: {server_label}[/dim]")
            return await original_call_tool(name, arguments)

        mcp.call_tool = instrumented_call_tool  # type: ignore[method-assign]

        with OpenRouterClient(api_key) as client:
            agent = Agent(
                client=client, model=_ORCHESTRATION_DEMO_MODEL, mcp_session=mcp
            )
            reply = await agent.run_async(_ORCHESTRATION_DEMO_PROMPT)

        # ── Финальный ответ ──────────────────────────────────────────────────
        console.print()
        console.print(
            Rule("[bold green]Финальный ответ агента[/bold green]", style="green")
        )
        console.print(Panel(reply, border_style="green", expand=False))

        # ── Показываем сохранённый файл ───────────────────────────────────
        import os
        from pathlib import Path

        report_file = Path(os.getcwd()) / "orchestration_report.md"
        if report_file.exists():
            content = report_file.read_text(encoding="utf-8")
            console.print()
            console.print(
                Rule("[bold blue]Сохранённый файл отчёта[/bold blue]", style="blue")
            )
            console.print(
                Panel(
                    content,
                    title=f"[dim]{report_file}[/dim]",
                    border_style="blue",
                    expand=False,
                )
            )

        # ── Итоговая статистика ──────────────────────────────────────────────
        console.print()
        stats_table = Table(
            title="Итог оркестрации",
            box=box.SIMPLE_HEAD,
            border_style="dim",
            header_style="bold",
        )
        stats_table.add_column("Параметр", style="dim")
        stats_table.add_column("Значение", style="bold cyan")
        stats_table.add_row("Серверов задействовано", "2")
        stats_table.add_row("Data & Analytics инструментов", str(len(data_tools)))
        stats_table.add_row("Tools & Storage инструментов", str(len(tools_tools)))
        stats_table.add_row("Шагов в пайплайне", "10")
        stats_table.add_row("Модель", _ORCHESTRATION_DEMO_MODEL)
        console.print(stats_table)


def run_orchestration_demo() -> None:
    """Длинный флоу оркестрации через два MCP-сервера с автоматической маршрутизацией."""
    from .config import ensure_config

    cfg = ensure_config()
    asyncio.run(_run_orchestration_demo(cfg.api_key))
