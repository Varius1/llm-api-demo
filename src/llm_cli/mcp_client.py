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
