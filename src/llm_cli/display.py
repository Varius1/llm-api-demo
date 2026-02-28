"""Rich-отрисовка: таблицы, панели, спиннеры, цветной вывод."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .models import BenchmarkResult, BranchInfo, ChatTurnStats, ModelConfig, StrategyType
from .strategy import STRATEGY_LABELS

console = Console()


def print_welcome(model: str, temperature: float | None) -> None:
    console.print()
    console.print(
        Panel(
            "[bold]LLM CLI Chat[/bold]\n"
            f"Модель: [cyan]{model}[/cyan]\n"
            f"Температура: [cyan]{temperature if temperature is not None else 'по умолчанию (0.2)'}[/cyan]\n\n"
            "Команды:\n"
            "  [yellow]/strategy sliding|facts|summary|branch[/yellow] — стратегия контекста\n"
            "  [yellow]/branch save [имя][/yellow] / [yellow]/branch switch <имя>[/yellow] / [yellow]/branch list[/yellow] — ветки\n"
            "  [yellow]/facts[/yellow] — показать KV-память (стратегия facts)\n"
            "  [yellow]/compress on|off[/yellow] — включить/выключить summary-сжатие\n"
            "  [yellow]/demo-strategies[/yellow] — сравнение 3 стратегий\n"
            "  [yellow]/demo-branch[/yellow] — демо веток: общий старт → 2 ветки → сравнение\n"
            "  [yellow]/demo-compare[/yellow] — сравнение off/on summary\n"
            "  [yellow]/temp 0.7[/yellow] — температура, [yellow]/model <id>[/yellow] — модель\n"
            "  [yellow]/overflow 9000[/yellow] — тест переполнения, [yellow]/clear[/yellow] — очистить историю\n"
            "  [yellow]exit[/yellow] — выход  |  двойной Enter — отправить сообщение",
            title="[bold cyan]LLM CLI[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()


def print_llm_response(text: str) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(text),
            title="[bold green]LLM[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def print_error(message: str) -> None:
    console.print(f"[bold red]Ошибка:[/bold red] {message}")
    console.print()


def print_chat_turn_stats(stats: ChatTurnStats) -> None:
    prompt = stats.prompt_tokens if stats.prompt_tokens is not None else "n/a"
    completion = (
        stats.completion_tokens if stats.completion_tokens is not None else "n/a"
    )
    total = stats.total_tokens if stats.total_tokens is not None else "n/a"
    turn_cost = (
        f"${stats.turn_cost_usd:.6f}" if stats.turn_cost_usd is not None else "n/a"
    )

    console.print(
        "[bold]Токены:[/bold] "
        f"запрос≈{stats.request_tokens_estimated}, "
        f"история≈{stats.history_tokens_estimated}, "
        f"prompt={prompt}, "
        f"response={completion}, "
        f"total={total}"
    )
    if (
        stats.raw_history_tokens_estimated is not None
        and stats.sent_history_tokens_estimated is not None
    ):
        console.print(
            "[dim]"
            f"Контекст до/после сжатия: {stats.raw_history_tokens_estimated} -> "
            f"{stats.sent_history_tokens_estimated} токенов (оценка)"
            "[/dim]"
        )
    strategy_label = STRATEGY_LABELS.get(StrategyType(stats.strategy), stats.strategy)
    console.print(f"[dim]Стратегия: {strategy_label}[/dim]")
    if stats.compression_enabled:
        state = "использован" if stats.used_summary else "не использован"
        console.print(
            "[dim]"
            f"Summary {state}, summary_chars={stats.summary_chars}, "
            f"сжато сообщений={stats.compressed_messages_count}"
            "[/dim]"
        )
    console.print(
        "[bold]Стоимость:[/bold] "
        f"за ход={turn_cost}, "
        f"сессия=${stats.session_cost_usd:.6f}"
    )
    console.print(
        "[dim]"
        f"Накопительно: prompt={stats.session_prompt_tokens}, "
        f"response={stats.session_completion_tokens}, "
        f"total={stats.session_total_tokens}"
        "[/dim]"
    )
    console.print()


def print_strategy_status(agent: object) -> None:
    """Вывести текущую стратегию и релевантные параметры агента."""
    from .agent import Agent  # локальный импорт для избежания цикла

    if not isinstance(agent, Agent):
        return

    strategy = agent.strategy
    label = STRATEGY_LABELS.get(strategy, strategy.value)
    status = agent.compression_status

    console.print(f"[dim]Стратегия: {label}[/dim]", end="")

    if strategy == StrategyType.SLIDING_WINDOW:
        console.print(f"[dim] | keep_last_n={status.keep_last_n}[/dim]")
    elif strategy == StrategyType.STICKY_FACTS:
        facts_count = len(agent.facts)
        console.print(f"[dim] | keep_last_n={status.keep_last_n} | фактов в памяти={facts_count}[/dim]")
    elif strategy == StrategyType.SUMMARY:
        console.print(
            f"[dim] | keep_last_n={status.keep_last_n} | "
            f"summary_chars={status.summary_chars} | "
            f"сжато={status.compressed_messages_count}[/dim]"
        )
    elif strategy == StrategyType.BRANCHING:
        branch = agent.current_branch or "—"
        branches_count = len(agent.branch_list())
        console.print(
            f"[dim] | текущая ветка={branch} | веток={branches_count}[/dim]"
        )
    else:
        console.print()
    console.print()


def print_branch_list(branches: list[BranchInfo], current: str | None) -> None:
    table = Table(title="Ветки диалога", show_lines=True)
    table.add_column("Имя", style="bold")
    table.add_column("Создана")
    table.add_column("Сообщений", justify="right")
    table.add_column("Активна", justify="center")

    for branch in branches:
        is_current = "✓" if branch.name == current else ""
        table.add_row(
            branch.name,
            branch.created_at,
            str(branch.messages_count),
            is_current,
        )
    console.print()
    console.print(table)
    console.print()


def print_benchmark_header(prompt: str, temperature: float | None) -> None:
    console.print()
    console.print(
        Panel(
            "[bold]СРАВНЕНИЕ МОДЕЛЕЙ OpenRouter[/bold]",
            border_style="bright_blue",
        )
    )
    console.print(f'Промпт: [italic]"{prompt}"[/italic]')
    temp_display = temperature if temperature is not None else "по умолчанию"
    console.print(f"Температура: [cyan]{temp_display}[/cyan]")
    console.print()


def print_model_header(model: ModelConfig) -> None:
    console.rule(f"[bold]{model.tier}: {model.display_name}[/bold]", style="yellow")
    console.print(f"  ID: [dim]{model.id}[/dim]")
    console.print(
        f"  Цена: [green]${model.input_price_per_million}/M[/green] вход | "
        f"[green]${model.output_price_per_million}/M[/green] выход"
    )
    console.print()


def print_model_result(result: BenchmarkResult) -> None:
    if result.error:
        console.print(f"  [red]Ошибка: {result.error}[/red]")
        console.print()
        return

    console.print(
        Panel(
            Markdown(result.response),
            title=f"[bold]{result.model.display_name}[/bold]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    secs = result.duration_ms / 1000
    console.print(f"  [bold]Время:[/bold] {result.duration_ms}мс ({secs:.1f}с)")
    if result.usage:
        u = result.usage
        console.print(
            f"  [bold]Токены:[/bold] {u.prompt_tokens} вход + "
            f"{u.completion_tokens} выход = {u.total_tokens} всего"
        )
    console.print(f"  [bold]Стоимость:[/bold] [green]${result.cost_usd:.6f}[/green]")
    console.print()


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    table = Table(
        title="СВОДНАЯ ТАБЛИЦА",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Модель", style="bold", min_width=18)
    table.add_column("Время", justify="right")
    table.add_column("Вход", justify="right")
    table.add_column("Выход", justify="right")
    table.add_column("Стоимость", justify="right", style="green")

    for r in results:
        if r.error:
            table.add_row(
                r.model.display_name,
                "[red]ОШИБКА[/red]",
                "-",
                "-",
                "-",
            )
        else:
            secs = r.duration_ms / 1000
            in_tok = str(r.usage.prompt_tokens) if r.usage else "-"
            out_tok = str(r.usage.completion_tokens) if r.usage else "-"
            table.add_row(
                r.model.display_name,
                f"{secs:.1f}с",
                in_tok,
                out_tok,
                f"${r.cost_usd:.6f}",
            )

    console.print()
    console.print(table)
    console.print()


def print_conclusions(results: list[BenchmarkResult]) -> None:
    successful = [r for r in results if r.error is None]
    if not successful:
        console.print("[dim]Нет успешных результатов для сравнения.[/dim]")
        return

    fastest = min(successful, key=lambda r: r.duration_ms)
    cheapest = min(successful, key=lambda r: r.cost_usd)
    longest = max(successful, key=lambda r: len(r.response))

    console.print("[bold]ВЫВОДЫ:[/bold]")
    console.rule(style="dim")
    console.print(
        f"  Быстрее всех: [bold]{fastest.model.display_name}[/bold] "
        f"({fastest.duration_ms / 1000:.1f}с)"
    )
    console.print(
        f"  Дешевле всех: [bold]{cheapest.model.display_name}[/bold] "
        f"(${cheapest.cost_usd:.6f})"
    )
    console.print(
        f"  Самый подробный: [bold]{longest.model.display_name}[/bold] "
        f"({len(longest.response)} символов)"
    )
    console.print()

    console.print("[bold]Ссылки на модели:[/bold]")
    for r in results:
        url = r.model.format_url()
        console.print(f"  [link={url}]{r.model.display_name}: {url}[/link]")
    console.print()


def print_judge_header(judge_name: str, judge_id: str) -> None:
    console.print(
        Panel(
            "[bold]ОЦЕНКА КАЧЕСТВА (модель-судья)[/bold]",
            border_style="magenta",
        )
    )
    console.print(f"  Судья: [bold]{judge_name}[/bold] ({judge_id})")
    console.print()


def print_judge_verdict(
    verdict: str,
    duration_ms: int,
    usage_prompt: int | None,
    usage_completion: int | None,
    usage_total: int | None,
    cost: float,
) -> None:
    console.print(
        Panel(
            Markdown(verdict),
            title="[bold magenta]Вердикт[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    secs = duration_ms / 1000
    console.print(f"  [bold]Время оценки:[/bold] {duration_ms}мс ({secs:.1f}с)")
    if usage_prompt is not None:
        console.print(
            f"  [bold]Токены судьи:[/bold] {usage_prompt} вход + "
            f"{usage_completion} выход = {usage_total}"
        )
    console.print(f"  [bold]Стоимость оценки:[/bold] [green]${cost:.6f}[/green]")
    console.print()
