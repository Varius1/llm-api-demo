"""Rich-отрисовка: таблицы, панели, спиннеры, цветной вывод."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .invariants import CATEGORY_LABELS, InvariantManager
from .memory import MemoryManager, UserProfile
from .models import BenchmarkResult, BranchInfo, ChatTurnStats, ModelConfig, StrategyType
from .strategy import STRATEGY_LABELS
from .task_fsm import (
    STAGE_CHAIN,
    STAGE_LABELS as FSM_STAGE_LABELS,
    ForbiddenTransitionError,
    TaskFSM,
    TaskStage,
)

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
            "  [bold cyan]Профиль и персонализация:[/bold cyan]\n"
            "  [yellow]/profile[/yellow] — показать активный профиль\n"
            "  [yellow]/profile new <имя>[/yellow] — создать профиль  |  [yellow]/profile switch <имя>[/yellow] — переключить\n"
            "  [yellow]/profile set <поле> <значение>[/yellow] — изменить поле  |  [yellow]/profile list[/yellow] — все профили\n"
            "  [yellow]/profile delete <имя>[/yellow] — удалить  |  [yellow]/profile reset[/yellow] — отключить профиль\n"
            "  [bold cyan]Task FSM (конечный автомат задачи):[/bold cyan]\n"
            "  [yellow]/task fsm start <имя>[/yellow] — запустить задачу (planning → execution → validation → done)\n"
            "  [yellow]/task fsm next [заметка][/yellow] — перейти к следующему этапу\n"
            "  [yellow]/task fsm pause[/yellow] — пауза (LLM отвечает свободно)  |  [yellow]/task fsm resume[/yellow] — возобновить\n"
            "  [yellow]/task fsm goto <этап>[/yellow] — попытка перехода (всегда заблокирована — показывает причину)\n"
            "  [yellow]/task fsm step <шаг>[/yellow] — текущий шаг  |  [yellow]/task fsm artifact <ключ> <текст>[/yellow] — артефакт\n"
            "  [yellow]/task fsm status[/yellow] — состояние FSM  |  [yellow]/task fsm clear[/yellow] — сброс\n"
            "  [yellow]/demo-fsm-lifecycle[/yellow] — [bold]для видео[/bold]: полный цикл с LLM + блокировки + пауза/resume (~5 запросов)\n"
            "  [yellow]/demo-fsm-guards[/yellow] — демо защиты переходов без API (чистая FSM-логика)\n"
            "  [yellow]/demo-fsm[/yellow] — расширенное демо: все этапы + сравнительная таблица ответов\n"
            "  [bold cyan]Модель памяти:[/bold cyan]\n"
            "  [yellow]/memory[/yellow] — показать все 3 слоя памяти\n"
            "  [yellow]/task <описание>[/yellow] — задать задачу в рабочей памяти  |  [yellow]/task clear[/yellow] — очистить\n"
            "  [yellow]/remember <текст>[/yellow] — заметка в долговременную память\n"
            "  [yellow]/remember <ключ>=<значение>[/yellow] — знание  |  [yellow]/remember profile <ключ>=<значение>[/yellow]\n"
            "  [yellow]/forget <ключ>[/yellow] — удалить из долговременной памяти\n"
            "  [yellow]/demo-persona[/yellow] — демо профилей: developer vs student, сравнение ответов\n"
            "  [yellow]/demo-memory[/yellow] — демо 3 слоёв памяти: хранение, влияние на ответы\n"
            "  [yellow]/demo-strategies[/yellow] — сравнение 3 стратегий\n"
            "  [yellow]/demo-branch[/yellow] — демо веток: общий старт → 2 ветки → сравнение\n"
            "  [yellow]/demo-compare[/yellow] — сравнение off/on summary\n"
            "  [bold cyan]Инварианты (неизменяемые правила проекта):[/bold cyan]\n"
            "  [yellow]/invariants[/yellow] — показать все инварианты\n"
            "  [yellow]/invariant-add <кат> <название> | <описание>[/yellow] — добавить  |  [yellow]/invariant-del <ID>[/yellow] — удалить\n"
            "  [yellow]/invariant-clear[/yellow] — очистить все  |  кат: [yellow]arch[/yellow] / [yellow]tech[/yellow] / [yellow]stack[/yellow] / [yellow]biz[/yellow]\n"
            "  [yellow]/demo-invariants[/yellow] — демо: инварианты + тест конфликта\n"
            "  [yellow]/model local[/yellow] — переключить на локальную LLM (llama.cpp)  |  [yellow]/model openrouter[/yellow] — переключить на OpenRouter\n"
            "  [yellow]/model <provider/model-id>[/yellow] — задать конкретную OpenRouter-модель\n"
            "  [yellow]/temp 0.7[/yellow] — температура\n"
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


def print_memory_state(
    memory: MemoryManager,
    recent_messages: list[object],
    keep_last_n: int = 3,
) -> None:
    """Отобразить все 3 слоя памяти в виде панелей."""
    from .models import ChatMessage  # локальный импорт для избежания цикла

    console.print()

    # ── Краткосрочная память ──────────────────────────────────────────────────
    dialog = [m for m in recent_messages if isinstance(m, ChatMessage) and m.role != "system"]
    tail = dialog[-keep_last_n:] if dialog else []
    if tail:
        lines = []
        for msg in tail:
            role_label = "[bold yellow]Вы[/bold yellow]" if msg.role == "user" else "[bold green]LLM[/bold green]"
            snippet = " ".join(msg.content.split())[:120]
            if len(msg.content) > 120:
                snippet += "..."
            lines.append(f"{role_label}: {snippet}")
        short_term_text = "\n".join(lines)
    else:
        short_term_text = "[dim]Диалог пуст[/dim]"

    console.print(
        Panel(
            short_term_text,
            title="[bold]Краткосрочная память[/bold] (последние сообщения текущего диалога)",
            border_style="yellow",
            padding=(0, 1),
        )
    )

    # ── Рабочая память ────────────────────────────────────────────────────────
    wm = memory.working
    if wm.is_empty():
        working_text = "[dim]Пусто. Используйте /task <описание> для установки задачи.[/dim]"
    else:
        wlines = []
        if wm.task:
            wlines.append(f"[bold]Задача:[/bold] {wm.task}")
        if wm.facts:
            facts_str = ", ".join(f"[cyan]{k}[/cyan]={v}" for k, v in wm.facts.items())
            wlines.append(f"[bold]Факты:[/bold] {facts_str}")
        if wm.goals:
            wlines.append("[bold]Цели:[/bold]")
            for i, g in enumerate(wm.goals, 1):
                wlines.append(f"  {i}. {g}")
        if wm.notes:
            wlines.append("[bold]Заметки:[/bold]")
            for note in wm.notes:
                wlines.append(f"  • {note}")
        working_text = "\n".join(wlines)
    console.print(
        Panel(
            working_text,
            title="[bold]Рабочая память[/bold] (данные текущей задачи, очищается при /clear)",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    # ── Долговременная память ─────────────────────────────────────────────────
    lt = memory.long_term
    if lt.is_empty():
        lt_text = (
            "[dim]Пусто. Используйте /remember для сохранения знаний и профиля.[/dim]"
        )
    else:
        ltlines = []
        if lt.profile:
            ltlines.append("[bold]Профиль:[/bold]")
            for k, v in lt.profile.items():
                ltlines.append(f"  [cyan]{k}[/cyan]: {v}")
        if lt.knowledge:
            ltlines.append("[bold]Знания:[/bold]")
            for k, v in lt.knowledge.items():
                ltlines.append(f"  [cyan]{k}[/cyan]: {v}")
        if lt.decisions:
            ltlines.append("[bold]Решения:[/bold]")
            for d in lt.decisions[-5:]:
                ltlines.append(f"  [{d.timestamp}] {d.text}")
        if lt.notes:
            ltlines.append("[bold]Заметки:[/bold]")
            for note in lt.notes[-5:]:
                ltlines.append(f"  • {note}")
        if lt.updated_at:
            ltlines.append(f"\n[dim]Обновлено: {lt.updated_at}[/dim]")
        lt_text = "\n".join(ltlines)
    console.print(
        Panel(
            lt_text,
            title="[bold]Долговременная память[/bold] (профиль и знания, НЕ очищается при /clear)",
            border_style="magenta",
            padding=(0, 1),
        )
    )
    console.print()


def print_profile(profile: UserProfile, name: str, *, active: bool = True) -> None:
    """Показать поля профиля пользователя в Rich-панели."""
    lines: list[str] = []
    field_map = [
        ("Имя", profile.name or "[dim](не задано)[/dim]"),
        ("Язык", profile.language),
        ("Стиль", profile.style),
        ("Формат", profile.format),
        ("Уровень", profile.expertise),
        ("Область", profile.domain or "[dim](не задано)[/dim]"),
    ]
    for label, value in field_map:
        lines.append(f"  [cyan]{label}:[/cyan] {value}")
    if profile.constraints:
        lines.append("  [cyan]Ограничения:[/cyan]")
        for c in profile.constraints:
            lines.append(f"    • {c}")
    active_mark = " [bold green](активный)[/bold green]" if active else ""
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold]Профиль: {name}[/bold]{active_mark}",
            border_style="green" if active else "dim",
            padding=(0, 1),
        )
    )


def print_profile_list(
    profiles: dict[str, UserProfile], active_name: str
) -> None:
    """Показать таблицу всех профилей."""
    if not profiles:
        console.print("[dim]Профили не созданы. Используйте /profile new <имя>[/dim]\n")
        return
    table = Table(title="Профили пользователя", show_lines=True)
    table.add_column("Имя", style="bold")
    table.add_column("Стиль")
    table.add_column("Формат")
    table.add_column("Уровень")
    table.add_column("Область")
    table.add_column("Активный", justify="center")
    for name, profile in profiles.items():
        is_active = "✓" if name == active_name else ""
        table.add_row(
            name,
            profile.style,
            profile.format,
            profile.expertise,
            profile.domain or "—",
            is_active,
        )
    console.print()
    console.print(table)
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


def print_task_fsm(fsm: TaskFSM) -> None:
    """Показать состояние FSM задачи в Rich-панели."""
    stage_colors: dict[TaskStage, str] = {
        TaskStage.PLANNING: "yellow",
        TaskStage.EXECUTION: "cyan",
        TaskStage.VALIDATION: "magenta",
        TaskStage.DONE: "green",
        TaskStage.PAUSED: "dim",
    }

    # ── Прогресс-строка этапов ────────────────────────────────────────────────
    progress_parts: list[str] = []
    for stage in STAGE_CHAIN:
        label = FSM_STAGE_LABELS[stage]
        color = stage_colors[stage]
        if fsm.stage == TaskStage.PAUSED and fsm.paused_at == stage.value:
            progress_parts.append(f"[bold {color}][{label} ⏸][/bold {color}]")
        elif stage == fsm.stage:
            progress_parts.append(f"[bold {color}][{label} ←][/bold {color}]")
        elif STAGE_CHAIN.index(stage) < (
            STAGE_CHAIN.index(fsm.stage)
            if fsm.stage in STAGE_CHAIN
            else len(STAGE_CHAIN)
        ):
            progress_parts.append(f"[{color}]✓ {label}[/{color}]")
        else:
            progress_parts.append(f"[dim]{label}[/dim]")
    progress_line = "  →  ".join(progress_parts)

    # ── Основная информация ───────────────────────────────────────────────────
    active_label = FSM_STAGE_LABELS[fsm.stage]
    active_color = stage_colors.get(fsm.stage, "white")
    lines: list[str] = [
        progress_line,
        "",
        f"[bold]Задача:[/bold] {fsm.task_name or '—'}",
        f"[bold]Этап:[/bold] [{active_color}]{active_label}[/{active_color}]",
    ]
    if fsm.stage == TaskStage.PAUSED and fsm.paused_at:
        paused_label = FSM_STAGE_LABELS.get(TaskStage(fsm.paused_at), fsm.paused_at)
        lines.append(f"[bold]Приостановлено на:[/bold] [dim]{paused_label}[/dim]")
    if fsm.current_step:
        lines.append(f"[bold]Текущий шаг:[/bold] {fsm.current_step}")
    if fsm.expected_action:
        lines.append(f"[bold]Ожидается:[/bold] {fsm.expected_action}")
    if fsm.created_at:
        lines.append(f"[dim]Создано: {fsm.created_at}[/dim]")

    # ── Артефакты ─────────────────────────────────────────────────────────────
    if fsm.artifacts:
        lines.append("")
        lines.append("[bold]Артефакты:[/bold]")
        for key, text in fsm.artifacts.items():
            preview = text[:80] + ("…" if len(text) > 80 else "")
            lines.append(f"  [cyan]{key}[/cyan]: {preview}")

    # ── История переходов ─────────────────────────────────────────────────────
    if fsm.transitions:
        lines.append("")
        lines.append("[bold]История переходов:[/bold]")
        for tr in fsm.transitions[-4:]:
            from_label = FSM_STAGE_LABELS.get(TaskStage(tr.from_stage), tr.from_stage)
            to_label = FSM_STAGE_LABELS.get(TaskStage(tr.to_stage), tr.to_stage)
            note_str = f" — {tr.note}" if tr.note else ""
            lines.append(
                f"  [dim]{tr.timestamp}[/dim]  {from_label} → {to_label}{note_str}"
            )

    title_color = active_color
    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold {title_color}]Task FSM: {fsm.task_name or 'задача'}[/bold {title_color}]",
            border_style=title_color,
            padding=(0, 1),
        )
    )
    console.print()


def print_fsm_transition_error(error: ForbiddenTransitionError) -> None:
    """Показать ошибку недопустимого перехода FSM в Rich-панели с красной рамкой."""
    from_label = FSM_STAGE_LABELS.get(error.from_stage, error.from_stage.value)
    to_label = FSM_STAGE_LABELS.get(error.to_stage, error.to_stage.value)

    lines = [
        f"[bold red]Переход запрещён:[/bold red]  «{from_label}» → «{to_label}»",
        "",
        f"[red]{error.reason}[/red]",
        "",
        f"[dim]Подсказка: {error.hint}[/dim]",
    ]
    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold red]Недопустимый переход FSM[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
    )
    console.print()


def print_invariants(manager: InvariantManager) -> None:
    """Вывести все инварианты, сгруппированные по категориям."""
    from .invariants import InvariantCategory

    invariants = manager.invariants
    if not invariants:
        console.print()
        console.print(
            Panel(
                "[dim]Инвариантов нет.[/dim]\n\n"
                "Добавьте: [yellow]/invariant-add <кат> <название> | <описание>[/yellow]\n"
                "Категории: [yellow]arch[/yellow] · [yellow]tech[/yellow] · [yellow]stack[/yellow] · [yellow]biz[/yellow]",
                title="[bold magenta]Инварианты[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        )
        console.print()
        return

    by_category: dict[str, list] = {}
    for inv in invariants:
        by_category.setdefault(inv.category.value, []).append(inv)

    lines: list[str] = []
    for cat_value in ("architecture", "technical", "stack", "business"):
        items = by_category.get(cat_value)
        if not items:
            continue
        label = CATEGORY_LABELS[cat_value]
        lines.append(f"[bold cyan]{label}[/bold cyan]")
        for inv in items:
            lines.append(f"  [yellow]{inv.id}[/yellow]  {inv.title}")
            lines.append(f"  [dim]       {inv.description}[/dim]")
            lines.append(f"  [dim]       Создан: {inv.created_at}[/dim]")
        lines.append("")

    total = len(invariants)
    console.print()
    console.print(
        Panel(
            "\n".join(lines).rstrip(),
            title=f"[bold magenta]Инварианты ({total})[/bold magenta]",
            border_style="magenta",
            padding=(0, 1),
        )
    )
    console.print()
