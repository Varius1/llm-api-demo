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
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
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

    console.print(Panel(
        "[bold cyan]Устанавливаю MCP-соединение...[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

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
            console.print(Rule("[bold magenta]Вызов инструментов[/bold magenta]", style="magenta"))
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
                    raw = content[0].text if hasattr(content[0], "text") else str(content[0])
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

    console.print(Panel(
        "[bold cyan]Запускаю агента с MCP-инструментами...[/bold cyan]\n"
        "[dim]Агент автоматически вызовет инструменты и вернёт результат[/dim]",
        border_style="cyan",
        expand=False,
    ))
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

        console.print(Rule("[bold magenta]Промпт агенту[/bold magenta]", style="magenta"))
        console.print(Panel(
            _AGENT_DEMO_PROMPT,
            border_style="dim",
            expand=False,
        ))
        console.print()
        console.print(Rule("[bold yellow]Tool Calling Loop[/bold yellow]", style="yellow"))
        console.print()

        with OpenRouterClient(api_key) as client:
            agent = Agent(client=client, model=_AGENT_DEMO_MODEL, mcp_session=mcp)
            reply = await agent.run_async(_AGENT_DEMO_PROMPT)

        console.print()
        console.print(Rule("[bold green]Финальный ответ LLM[/bold green]", style="green"))
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

    console.print(Panel(
        "[bold cyan]MCP Pipeline Demo[/bold cyan]\n"
        "[dim]search → summarize → save_to_file[/dim]\n\n"
        "[white]Агент автоматически выполнит цепочку инструментов:[/white]\n"
        "  [yellow]1.[/yellow] [bold]search[/bold]       — получить данные\n"
        "  [yellow]2.[/yellow] [bold]summarize[/bold]    — обработать / сжать\n"
        "  [yellow]3.[/yellow] [bold]save_to_file[/bold] — сохранить результат",
        border_style="cyan",
        expand=False,
    ))
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
        console.print(f"[dim]+ ещё {other_tools_count} инструментов зарегистрировано на сервере[/dim]\n")

        console.print(Rule("[bold magenta]Промпт агенту[/bold magenta]", style="magenta"))
        console.print(Panel(
            _PIPELINE_DEMO_PROMPT,
            border_style="dim",
            expand=False,
        ))
        console.print()
        console.print(Rule("[bold yellow]Tool Calling Pipeline[/bold yellow]", style="yellow"))
        console.print()

        with OpenRouterClient(api_key) as client:
            agent = Agent(client=client, model=_PIPELINE_DEMO_MODEL, mcp_session=mcp)
            reply = await agent.run_async(_PIPELINE_DEMO_PROMPT)

        console.print()
        console.print(Rule("[bold green]Финальный ответ LLM[/bold green]", style="green"))
        console.print(Panel(reply, border_style="green", expand=False))

        import os
        from pathlib import Path
        result_file = Path(os.getcwd()) / "pipeline_result.txt"
        if result_file.exists():
            content = result_file.read_text(encoding="utf-8")
            console.print()
            console.print(Rule("[bold blue]Содержимое сохранённого файла[/bold blue]", style="blue"))
            console.print(Panel(
                content,
                title=f"[dim]{result_file}[/dim]",
                border_style="blue",
                expand=False,
            ))


def run_pipeline_demo() -> None:
    """Автоматический пайплайн: search → summarize → save_to_file."""
    from .config import ensure_config

    cfg = ensure_config()
    asyncio.run(_run_pipeline_demo(cfg.api_key))
