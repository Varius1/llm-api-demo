"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

from rich.console import Console

from .agent import Agent
from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import print_error, print_llm_response, print_welcome
from .models import BENCHMARK_PROMPT

console = Console()


def run_chat(client: OpenRouterClient, cfg: AppConfig) -> None:
    agent = Agent(
        client=client,
        model=cfg.default_model,
        temperature=cfg.temperature,
    )
    benchmark_prompt = cfg.benchmark_prompt or BENCHMARK_PROMPT

    print_welcome(agent.model, agent.temperature)
    if agent.restored_messages_count > 0:
        console.print(
            f"[dim]Восстановлено сообщений из истории: {agent.restored_messages_count}[/dim]"
        )
        console.print()

    while True:
        try:
            lines = _read_multiline_input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]До свидания![/dim]")
            break

        if lines is None:
            continue

        text = lines.strip()
        if not text:
            continue

        if text.lower() in ("exit", "quit"):
            console.print("[dim]До свидания![/dim]")
            break

        if text.startswith("/temp"):
            _handle_temp(text, agent)
            continue

        if text.startswith("/model"):
            _handle_model(text, agent)
            continue

        if text == "/compare":
            console.print()
            run_benchmark(client, benchmark_prompt, cfg.models, agent.temperature)
            continue

        if text == "/clear":
            agent.clear_history()
            console.print("[green]История очищена.[/green]\n")
            continue

        try:
            with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
                reply = agent.run(text)
            print_llm_response(reply)
        except Exception as e:
            print_error(str(e))


def _read_multiline_input() -> str | None:
    """Читает многострочный ввод. Двойной Enter — отправить."""
    console.print("[bold yellow]Вы:[/bold yellow] ", end="")
    lines: list[str] = []
    empty_count = 0

    while True:
        try:
            line = input()
        except EOFError:
            if lines:
                break
            raise

        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)

    return "\n".join(lines).strip() or None


def _handle_temp(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/temp").strip()
    if not raw:
        current = agent.temperature
        console.print(
            f"[dim]Текущая температура: {current if current is not None else 'по умолчанию (1.0)'}[/dim]"
        )
        return
    try:
        new_temp = float(raw)
        agent.temperature = new_temp
        console.print(f"[green]Температура: {new_temp}[/green]\n")
    except ValueError:
        print_error(f"Некорректное значение температуры: {raw}")


def _handle_model(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/model").strip()
    if not raw:
        console.print(f"[dim]Текущая модель: {agent.model}[/dim]")
        return
    agent.model = raw
    console.print(f"[green]Модель: {raw}[/green]\n")
