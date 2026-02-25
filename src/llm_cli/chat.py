"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

from rich.console import Console

from .agent import Agent
from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import print_chat_turn_stats, print_error, print_llm_response, print_welcome
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

        transforms: list[str] | None = None
        if text.startswith("/overflow"):
            payload = _build_overflow_payload(text)
            if payload is None:
                continue
            text = payload
            # Для гарантии реального переполнения отключаем авто-трансформации OpenRouter.
            transforms = []

        try:
            with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
                reply, stats = agent.run_with_stats(text, transforms=transforms)
            print_llm_response(reply)
            print_chat_turn_stats(stats)
        except Exception as e:
            raw_error = str(e)
            if _looks_like_context_overflow(raw_error):
                print_error(
                    "Контекст модели переполнен (лимит токенов превышен). "
                    "Очистите историю командой /clear или смените модель на более длинный контекст.\n"
                    f"Детали API: {raw_error}"
                )
            else:
                print_error(raw_error)


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


def _build_overflow_payload(text: str) -> str | None:
    raw = text.removeprefix("/overflow").strip()
    if not raw:
        target_tokens = 9000
    else:
        try:
            target_tokens = int(raw)
        except ValueError:
            print_error("Использование: /overflow <целевые-токены>, например /overflow 9000")
            return None

    if target_tokens <= 0:
        print_error("Количество токенов для /overflow должно быть положительным.")
        return None

    payload = _generate_large_prompt(target_tokens)
    estimated_tokens = max(1, (len(payload) + 3) // 4)
    console.print(
        f"[yellow]Отправляется тестовый запрос для переполнения:[/yellow] "
        f"{estimated_tokens} токенов (оценка), режим strict (transforms: [])."
    )
    console.print()
    return payload


def _generate_large_prompt(target_tokens: int) -> str:
    # Оценка: 1 токен ~= 4 символа.
    target_chars = target_tokens * 4
    chunk = (
        "Это демонстрация переполнения контекста. "
        "Сохраняй последовательность букв ABCDEFGHIJKLMNOPQRSTUVWXYZ. "
    )
    repeated = (chunk * ((target_chars // len(chunk)) + 1))[:target_chars]
    return (
        "Служебный тест переполнения контекста. Не отвечай кратко, сначала прочитай весь блок.\n\n"
        + repeated
    )


def _looks_like_context_overflow(error_text: str) -> bool:
    normalized = error_text.lower()
    if "http 400" not in normalized:
        return False
    keywords = (
        "context",
        "token",
        "maximum",
        "max context",
        "context_length",
        "input too long",
    )
    return any(key in normalized for key in keywords)
