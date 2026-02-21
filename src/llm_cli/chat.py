"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import print_error, print_llm_response, print_welcome
from .models import BENCHMARK_PROMPT, ChatMessage

console = Console()


def run_chat(client: OpenRouterClient, cfg: AppConfig) -> None:
    model = cfg.default_model
    temperature: float | None = cfg.temperature
    benchmark_prompt = cfg.benchmark_prompt or BENCHMARK_PROMPT

    print_welcome(model, temperature)

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
            temperature = _handle_temp(text, temperature)
            continue

        if text == "/compare":
            console.print()
            run_benchmark(client, benchmark_prompt, cfg.models, temperature)
            continue

        if text.startswith("/model"):
            model = _handle_model(text, model)
            continue

        messages = [ChatMessage(role="user", content=text)]

        try:
            with console.status(
                "[bold cyan]Думаю...[/bold cyan]",
                spinner="dots",
            ):
                reply = client.send(messages, model, temperature)
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


def _handle_temp(text: str, current: float | None) -> float | None:
    raw = text.removeprefix("/temp").strip()
    if not raw:
        console.print(
            f"[dim]Текущая температура: {current if current is not None else 'по умолчанию (1.0)'}[/dim]"
        )
        return current
    try:
        new_temp = float(raw)
        console.print(f"[green]Температура: {new_temp}[/green]")
        console.print()
        return new_temp
    except ValueError:
        print_error(f"Некорректное значение температуры: {raw}")
        return current


def _handle_model(text: str, current: str) -> str:
    raw = text.removeprefix("/model").strip()
    if not raw:
        console.print(f"[dim]Текущая модель: {current}[/dim]")
        return current
    console.print(f"[green]Модель: {raw}[/green]")
    console.print()
    return raw
