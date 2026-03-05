"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import Agent
from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import (
    print_branch_list,
    print_chat_turn_stats,
    print_error,
    print_invariants,
    print_llm_response,
    print_memory_state,
    print_profile,
    print_profile_list,
    print_strategy_status,
    print_task_fsm,
    print_welcome,
)
from .memory import PROFILE_EXPERTISE_OPTIONS, PROFILE_FORMAT_OPTIONS, PROFILE_STYLE_OPTIONS, UserProfile
from .models import BENCHMARK_PROMPT, StrategyType
from .strategy import STRATEGY_LABELS

console = Console()

DEMO_FACT_MESSAGES = [
    "Запомни: проект называется Atlas CRM.",
    "Клиент: NorthWind Logistics.",
    "Дедлайн релиза: 30 апреля.",
    "Бюджет: 1.8 млн рублей.",
    "Команда: Анна (PM), Илья (backend), Мария (frontend).",
    "Стек: FastAPI, PostgreSQL, React.",
    "Критичный риск: задержка интеграции с 1С.",
    "Приоритет №1: модуль заявок.",
    "Приоритет №2: отчёты по SLA.",
    "Ограничение: никаких платных SaaS.",
    "Требование: логирование действий пользователей 180 дней.",
    "Канал связи с клиентом: Telegram + еженедельный Zoom.",
    "План релиза: beta 15 апреля, production 30 апреля.",
    "Интеграция: первыми подключаем CRM-формы сайта и 1С.",
    "KPI: время обработки заявки до 30 минут.",
    "KPI: SLA 95% по обращениям первого приоритета.",
    "Нефункционально: аудит всех изменений критичных полей.",
    "Безопасность: доступ по ролям admin/manager/operator.",
    "Отчёты: ежедневный дашборд по просроченным тикетам.",
    "Требование: экспорт отчётов в CSV и XLSX.",
    "UX: карточка заявки должна открываться < 1 секунды.",
    "Риск: отпуск ключевого backend-разработчика в апреле.",
    "Ограничение: деплой только в инфраструктуру клиента.",
    "Ограничение: PostgreSQL версия не выше 14.",
    "Коммуникации: демо заказчику каждую пятницу.",
    "Бизнес-правило: обязательный источник лида в каждой заявке.",
    "Бизнес-правило: нельзя закрывать заявку без причины закрытия.",
    "Процесс: приоритет P1 эскалируется через 15 минут.",
]

DEMO_QUESTIONS = [
    "Сделай краткое резюме проекта в 5 пунктах. Ответь строго на русском, кратко.",
    "Назови всех участников команды и их роли. Ответь строго на русском, кратко.",
    "Какие 3 ключевых ограничения/риска ты помнишь? Ответь строго на русском, кратко.",
]

DEMO_QUALITY_KEYWORDS = [
    ["atlas", "crm", "northwind", "30", "апрел", "1.8", "fastapi", "postgresql", "react"],
    ["анна", "pm", "илья", "backend", "мария", "frontend"],
    ["риск", "1с", "saas", "логирован", "180", "telegram", "zoom"],
]


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
    print_strategy_status(agent)

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

        if text.startswith("/compress"):
            _handle_compress(text, agent)
            continue

        if text.startswith("/strategy"):
            _handle_strategy(text, agent)
            continue

        if text.startswith("/branch"):
            _handle_branch(text, agent)
            continue

        if text == "/facts":
            _print_facts(agent)
            continue

        if text == "/memory":
            print_memory_state(agent.memory, agent.history)
            continue

        if text.startswith("/profile"):
            _handle_profile(text, agent)
            continue

        if text == "/demo-persona":
            _run_demo_persona(agent)
            continue

        if text == "/demo-fsm":
            _run_demo_fsm(agent)
            continue

        if text.startswith("/task"):
            _handle_task(text, agent)
            continue

        if text.startswith("/remember"):
            _handle_remember(text, agent)
            continue

        if text.startswith("/forget"):
            _handle_forget(text, agent)
            continue

        if text == "/compare":
            console.print()
            run_benchmark(client, benchmark_prompt, cfg.models, agent.temperature)
            continue

        if text == "/demo-compare":
            _run_demo_compare(agent)
            continue

        if text == "/demo-strategies":
            _run_demo_strategies(agent)
            continue

        if text == "/demo-branch":
            _run_demo_branch(agent)
            continue

        if text == "/demo-memory":
            _run_demo_memory(agent)
            continue

        if text == "/invariants":
            print_invariants(agent.invariants)
            continue

        if text.startswith("/invariant-add"):
            _handle_invariant_add(text, agent)
            continue

        if text.startswith("/invariant-del"):
            _handle_invariant_del(text, agent)
            continue

        if text == "/invariant-clear":
            agent.invariants.clear()
            console.print("[green]Все инварианты удалены.[/green]\n")
            continue

        if text == "/demo-invariants":
            _run_demo_invariants(agent)
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
            f"[dim]Текущая температура: {current if current is not None else 'по умолчанию (0.2)'}[/dim]"
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


def _handle_compress(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/compress").strip().lower()
    if not raw:
        print_strategy_status(agent)
        return

    if raw in ("on", "1", "true", "yes"):
        agent.compression_enabled = True
        console.print("[green]Компрессия контекста: ON (стратегия: summary)[/green]\n")
        print_strategy_status(agent)
        return

    if raw in ("off", "0", "false", "no"):
        agent.compression_enabled = False
        console.print("[yellow]Компрессия контекста: OFF (стратегия: sliding)[/yellow]\n")
        print_strategy_status(agent)
        return

    print_error("Использование: /compress on|off")


def _handle_strategy(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/strategy").strip().lower()
    if not raw:
        print_strategy_status(agent)
        return

    parts = raw.split()
    strategy_name = parts[0]
    extra = parts[1] if len(parts) > 1 else None

    strategy_map = {
        "sliding": StrategyType.SLIDING_WINDOW,
        "window": StrategyType.SLIDING_WINDOW,
        "facts": StrategyType.STICKY_FACTS,
        "kv": StrategyType.STICKY_FACTS,
        "summary": StrategyType.SUMMARY,
        "compress": StrategyType.SUMMARY,
        "branch": StrategyType.BRANCHING,
        "branching": StrategyType.BRANCHING,
    }

    if strategy_name not in strategy_map:
        print_error(
            f"Неизвестная стратегия: «{strategy_name}».\n"
            "Доступные: sliding, facts, summary, branch"
        )
        return

    new_strategy = strategy_map[strategy_name]

    # Необязательный параметр keep_n для sliding.
    if extra and new_strategy == StrategyType.SLIDING_WINDOW:
        try:
            agent._keep_last_n = max(1, int(extra))
        except ValueError:
            print_error(f"Некорректное значение keep_n: {extra}")
            return

    agent.strategy = new_strategy
    label = STRATEGY_LABELS[new_strategy]
    console.print(f"[green]Стратегия: {label}[/green]\n")
    print_strategy_status(agent)


def _handle_branch(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/branch").strip()
    parts = raw.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "save":
        name = arg or "main"
        agent.branch_save(name)
        console.print(f"[green]Ветка «{name}» сохранена ({len(agent.history)} сообщений).[/green]\n")
        return

    if sub == "switch":
        if not arg:
            print_error("Использование: /branch switch <имя>")
            return
        try:
            agent.branch_switch(arg)
            console.print(f"[green]Переключено на ветку «{arg}».[/green]\n")
        except ValueError as e:
            print_error(str(e))
        return

    if sub == "list":
        branches = agent.branch_list()
        if not branches:
            console.print("[dim]Нет сохранённых веток.[/dim]\n")
        else:
            print_branch_list(branches, agent.current_branch)
        return

    if not sub:
        branches = agent.branch_list()
        if not branches:
            console.print("[dim]Нет сохранённых веток.[/dim]\n")
        else:
            print_branch_list(branches, agent.current_branch)
        return

    print_error("Использование: /branch save [имя] | /branch switch <имя> | /branch list")


def _print_facts(agent: Agent) -> None:
    facts = agent.facts
    if not facts:
        console.print("[dim]KV-память пуста. Переключитесь на стратегию facts (/strategy facts).[/dim]\n")
        return
    console.print("[bold]Текущие факты (KV-память):[/bold]")
    for key, value in facts.items():
        console.print(f"  [cyan]{key}[/cyan]: {value}")
    console.print()


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


def _run_demo_compare(agent: Agent) -> None:
    """Автодемо: запускает одинаковый сценарий в режимах off/on и печатает сравнение."""
    original_compression = agent.compression_enabled
    results: list[dict[str, object]] = []

    console.print(
        "[bold cyan]Демо-сравнение запущено:[/bold cyan] "
        "сначала без сжатия, потом со сжатием."
    )
    console.print("[dim]Это выполнит много реальных API-запросов и увеличит расход токенов.[/dim]\n")

    for mode_label, compression_enabled in (
        ("Без сжатия", False),
        ("Со сжатием", True),
    ):
        agent.clear_history()
        agent.reset_session_metrics()
        agent.compression_enabled = compression_enabled
        answers: list[str] = []
        last_stats = None
        total_steps = len(DEMO_FACT_MESSAGES) + len(DEMO_QUESTIONS)
        step_no = 0
        (
            prompt_before,
            completion_before,
            total_before,
            cost_before,
        ) = agent.get_session_totals()

        console.rule(f"[bold]{mode_label}[/bold]", style="cyan")
        mode_error: str | None = None
        try:
            for message in DEMO_FACT_MESSAGES:
                step_no += 1
                _print_demo_step_request(mode_label, step_no, total_steps, message, "fact")
                with console.status(
                    f"[bold cyan]{mode_label}: отправка факта {step_no}/{total_steps}...[/bold cyan]",
                    spinner="dots",
                ):
                    _, last_stats = _run_demo_request_with_retry(agent, message)
            for question in DEMO_QUESTIONS:
                step_no += 1
                _print_demo_step_request(
                    mode_label, step_no, total_steps, question, "question"
                )
                with console.status(
                    f"[bold cyan]{mode_label}: контрольный вопрос {step_no}/{total_steps}...[/bold cyan]",
                    spinner="dots",
                ):
                    answer, last_stats = _run_demo_request_with_retry(agent, question)
                answers.append(answer)
        except Exception as e:
            mode_error = str(e)
            print_error(f"{mode_label}: автопрогон прерван из-за ошибки API.\n{mode_error}")

        quality_score, breakdown = _score_quality(answers) if answers else (0, [0, 0, 0])
        (
            prompt_after,
            completion_after,
            total_after,
            cost_after,
        ) = agent.get_session_totals()
        results.append(
            {
                "mode": mode_label,
                "compression": compression_enabled,
                "answers": answers,
                "error": mode_error,
                "quality": quality_score,
                "breakdown": breakdown,
                "prompt_tokens": max(0, prompt_after - prompt_before),
                "completion_tokens": max(0, completion_after - completion_before),
                "total_tokens": max(0, total_after - total_before),
                "cost_usd": max(0.0, cost_after - cost_before),
            }
        )

    agent.compression_enabled = original_compression
    _print_demo_compare_results(results)


def _score_quality(answers: list[str]) -> tuple[int, list[int]]:
    scores: list[int] = []
    for i, answer in enumerate(answers):
        answer_l = answer.lower()
        hits = sum(1 for keyword in DEMO_QUALITY_KEYWORDS[i] if keyword in answer_l)
        # Нормируем в шкалу 0..5.
        max_hits = max(1, len(DEMO_QUALITY_KEYWORDS[i]))
        score = round((hits / max_hits) * 5)
        scores.append(score)
    return sum(scores), scores


def _print_demo_compare_results(results: list[dict[str, object]]) -> None:
    console.print()
    table = Table(title="DEMO COMPARE: без сжатия vs со сжатием", show_lines=True)
    table.add_column("Режим", style="bold")
    table.add_column("Качество (0-15)", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Стоимость", justify="right")

    for result in results:
        breakdown = result["breakdown"]
        mode_value = str(result["mode"])
        if result.get("error"):
            mode_value = f"{mode_value} [ERR]"
        table.add_row(
            mode_value,
            f"{result['quality']} ({breakdown[0]}/{breakdown[1]}/{breakdown[2]})",
            str(result["prompt_tokens"]),
            str(result["completion_tokens"]),
            str(result["total_tokens"]),
            f"${float(result['cost_usd']):.6f}",
        )

    console.print(table)
    console.print()

    for index, question in enumerate(DEMO_QUESTIONS, start=1):
        console.rule(f"[bold]Контрольный вопрос {index}[/bold]", style="dim")
        console.print(f"[bold]Вопрос:[/bold] {question}")
        for result in results:
            answers = result["answers"]
            if len(answers) >= index:
                snippet = _one_line(str(answers[index - 1]))[:280]
            else:
                snippet = "(нет ответа — прогон прерван из-за ошибки API)"
            console.print(f"[bold]{result['mode']}:[/bold] {snippet}")
        console.print()

    if len(results) == 2 and not results[0].get("error") and not results[1].get("error"):
        off = results[0]
        on = results[1]
        off_total = int(off["total_tokens"])
        on_total = int(on["total_tokens"])
        off_cost = float(off["cost_usd"])
        on_cost = float(on["cost_usd"])
        token_delta_pct = ((on_total - off_total) / off_total * 100) if off_total else 0.0
        cost_delta_pct = ((on_cost - off_cost) / off_cost * 100) if off_cost else 0.0
        if on_total <= off_total and on_cost <= off_cost:
            console.print(
                f"[bold green]Экономия со сжатием:[/bold green] "
                f"токены {off_total} -> {on_total} ({-token_delta_pct:.1f}% экономии), "
                f"стоимость ${off_cost:.6f} -> ${on_cost:.6f} ({-cost_delta_pct:.1f}% экономии)"
            )
        else:
            console.print(
                f"[bold yellow]Со сжатием вырос расход:[/bold yellow] "
                f"токены {off_total} -> {on_total} ({token_delta_pct:.1f}% к расходу), "
                f"стоимость ${off_cost:.6f} -> ${on_cost:.6f} ({cost_delta_pct:.1f}% к расходу)"
            )
        console.print()


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _print_demo_step_request(
    mode_label: str, step_no: int, total_steps: int, text: str, kind: str
) -> None:
    kind_label = "факт" if kind == "fact" else "контрольный вопрос"
    snippet = _one_line(text)
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    console.print(
        f"[dim][{mode_label}] {step_no}/{total_steps} | {kind_label} -> {snippet}[/dim]"
    )


def _run_demo_request_with_retry(
    agent: Agent, text: str, max_attempts: int = 3
) -> tuple[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return agent.run_with_stats(text)
        except Exception as e:
            last_error = e
            if attempt == max_attempts or not _is_transient_api_error(str(e)):
                raise
            backoff = 1.0 * attempt
            console.print(
                f"[yellow]Временная ошибка API, повтор {attempt}/{max_attempts} через {backoff:.0f}с...[/yellow]"
            )
            time.sleep(backoff)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Неизвестная ошибка demo-запроса")


def _is_transient_api_error(error_text: str) -> bool:
    normalized = error_text.lower()
    transient_markers = (
        "http 500",
        "http 502",
        "http 503",
        "http 429",
        "http 408",
        "timeout",
        "timed out",
    )
    return any(marker in normalized for marker in transient_markers)


def _run_demo_strategies(agent: Agent) -> None:
    """Автодемо: прогоняет одинаковый сценарий для 3 стратегий и выводит сравнение."""
    original_strategy = agent.strategy
    strategies_to_run = [
        ("Sliding Window", StrategyType.SLIDING_WINDOW),
        ("Sticky Facts",   StrategyType.STICKY_FACTS),
        ("Summary",        StrategyType.SUMMARY),
    ]
    results: list[dict[str, object]] = []

    console.print(
        "[bold cyan]Demo-Strategies запущено:[/bold cyan] "
        "3 стратегии × один сценарий."
    )
    console.print(
        "[dim]Будет выполнено много API-запросов. Нажмите Ctrl+C для отмены.[/dim]\n"
    )

    total_steps = len(DEMO_FACT_MESSAGES) + len(DEMO_QUESTIONS)

    for mode_label, strategy in strategies_to_run:
        agent.clear_history()
        agent.reset_session_metrics()
        agent.strategy = strategy
        answers: list[str] = []
        step_no = 0
        (prompt_before, completion_before, total_before, cost_before) = agent.get_session_totals()

        console.rule(f"[bold]{mode_label}[/bold]", style="cyan")
        mode_error: str | None = None
        try:
            for message in DEMO_FACT_MESSAGES:
                step_no += 1
                _print_demo_step_request(mode_label, step_no, total_steps, message, "fact")
                with console.status(
                    f"[cyan]{mode_label}: факт {step_no}/{total_steps}...[/cyan]",
                    spinner="dots",
                ):
                    _run_demo_request_with_retry(agent, message)

            for question in DEMO_QUESTIONS:
                step_no += 1
                _print_demo_step_request(mode_label, step_no, total_steps, question, "question")
                with console.status(
                    f"[cyan]{mode_label}: вопрос {step_no}/{total_steps}...[/cyan]",
                    spinner="dots",
                ):
                    answer, _ = _run_demo_request_with_retry(agent, question)
                answers.append(answer)
        except Exception as e:
            mode_error = str(e)
            print_error(f"{mode_label}: прерван из-за ошибки API.\n{mode_error}")

        quality_score, breakdown = _score_quality(answers) if answers else (0, [0, 0, 0])
        (prompt_after, completion_after, total_after, cost_after) = agent.get_session_totals()
        results.append({
            "mode": mode_label,
            "strategy": strategy.value,
            "answers": answers,
            "error": mode_error,
            "quality": quality_score,
            "breakdown": breakdown,
            "prompt_tokens": max(0, prompt_after - prompt_before),
            "completion_tokens": max(0, completion_after - completion_before),
            "total_tokens": max(0, total_after - total_before),
            "cost_usd": max(0.0, cost_after - cost_before),
        })

    agent.strategy = original_strategy
    _print_demo_strategies_results(results)


def _print_demo_strategies_results(results: list[dict[str, object]]) -> None:
    console.print()
    table = Table(title="DEMO STRATEGIES: сравнение стратегий контекста", show_lines=True)
    table.add_column("Стратегия", style="bold")
    table.add_column("Качество (0-15)", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Стоимость", justify="right")

    for result in results:
        mode_value = str(result["mode"])
        if result.get("error"):
            mode_value = f"{mode_value} [ERR]"
        breakdown = result["breakdown"]
        table.add_row(
            mode_value,
            f"{result['quality']} ({breakdown[0]}/{breakdown[1]}/{breakdown[2]})",
            str(result["prompt_tokens"]),
            str(result["completion_tokens"]),
            str(result["total_tokens"]),
            f"${float(result['cost_usd']):.6f}",
        )
    console.print(table)
    console.print()

    for index, question in enumerate(DEMO_QUESTIONS, start=1):
        console.rule(f"[bold]Контрольный вопрос {index}[/bold]", style="dim")
        console.print(f"[bold]Вопрос:[/bold] {question}")
        for result in results:
            answers = result["answers"]
            if isinstance(answers, list) and len(answers) >= index:
                snippet = _one_line(str(answers[index - 1]))[:280]
            else:
                snippet = "(нет ответа — прогон прерван)"
            console.print(f"[bold]{result['mode']}:[/bold] {snippet}")
        console.print()

    # Резюме по токенам (сравнение с первым результатом = sliding window).
    if len(results) >= 2 and not any(r.get("error") for r in results):
        baseline = results[0]
        base_total = int(baseline["total_tokens"])
        base_cost = float(baseline["cost_usd"])
        console.print("[bold]Сравнение токенов vs Sliding Window:[/bold]")
        for result in results[1:]:
            r_total = int(result["total_tokens"])
            r_cost = float(result["cost_usd"])
            delta_tok = r_total - base_total
            delta_pct = (delta_tok / base_total * 100) if base_total else 0.0
            sign = "+" if delta_tok >= 0 else ""
            console.print(
                f"  {result['mode']}: токены {base_total} -> {r_total} "
                f"({sign}{delta_pct:.1f}%), стоимость ${base_cost:.6f} -> ${r_cost:.6f}"
            )
        console.print()


# ─── Сценарий для demo-branch ──────────────────────────────────────────────

# Общие факты, которые войдут в обе ветки.
DEMO_BRANCH_COMMON_FACTS = [
    "Запомни: проект называется Atlas CRM.",
    "Клиент: NorthWind Logistics.",
    "Дедлайн релиза: 30 апреля.",
    "Бюджет: 1.8 млн рублей.",
    "Команда: Анна (PM), Илья (backend), Мария (frontend).",
    "Стек: FastAPI, PostgreSQL, React.",
    "Ограничение: PostgreSQL версия не выше 14.",
    "Ограничение: деплой только в инфраструктуру клиента.",
    "Ограничение: никаких платных SaaS.",
    "Критичный риск: задержка интеграции с 1С.",
]

# Дополнительные факты только для ветки «main» (фокус: риски/технический долг).
DEMO_BRANCH_MAIN_FACTS = [
    "Риск: отпуск ключевого backend-разработчика Ильи в апреле.",
    "Риск: нет автоматических тестов на модуль интеграции 1С.",
    "Технический долг: модуль импорта данных написан без документации.",
]

# Дополнительные факты только для ветки «alt» (фокус: фичи и UX).
DEMO_BRANCH_ALT_FACTS = [
    "Приоритет №1: модуль заявок — должен работать быстрее 1 секунды.",
    "UX-требование: карточка заявки открывается < 1 секунды.",
    "Новая фича по запросу клиента: экспорт отчётов в CSV и XLSX.",
]

# Контрольные вопросы для каждой ветки.
DEMO_BRANCH_QUESTIONS = [
    "Назови 3 главных риска проекта. Ответь строго на русском, кратко.",
    "Назови состав команды и их роли. Ответь строго на русском, кратко.",
    "Какие ключевые ограничения ты помнишь? Ответь строго на русском, кратко.",
]


def _run_demo_branch(agent: Agent) -> None:
    """Демо стратегии Branching: общий старт → 2 независимые ветки → сравнение."""
    original_strategy = agent.strategy
    original_branch = agent.current_branch

    console.print(
        "[bold cyan]Demo-Branch запущено.[/bold cyan] "
        "Общие факты → checkpoint → ветка «main» (риски) и «alt» (фичи) → сравнение."
    )
    console.print("[dim]Будет выполнено несколько API-запросов.[/dim]\n")

    # Шаг 1: подготовка — очистить историю и переключить стратегию.
    agent.clear_history()
    agent.reset_session_metrics()
    agent.strategy = StrategyType.BRANCHING

    total_common = len(DEMO_BRANCH_COMMON_FACTS)
    console.rule("[bold]Шаг 1: общий контекст (войдёт в обе ветки)[/bold]", style="cyan")

    common_error: str | None = None
    for i, fact in enumerate(DEMO_BRANCH_COMMON_FACTS, start=1):
        console.print(f"[dim][общий] {i}/{total_common} -> {fact}[/dim]")
        with console.status(f"[cyan]отправка факта {i}/{total_common}...[/cyan]", spinner="dots"):
            try:
                _run_demo_request_with_retry(agent, fact)
            except Exception as e:
                common_error = str(e)
                print_error(f"Ошибка на шаге 1: {common_error}")
                break

    if common_error:
        agent.strategy = original_strategy
        return

    # Шаг 2: сохранить checkpoint как «main» и «alt».
    console.rule("[bold]Шаг 2: сохраняем checkpoint[/bold]", style="cyan")
    agent.branch_save("main")
    agent.branch_save("alt")
    console.print("[green]Checkpoint сохранён как ветки «main» и «alt».[/green]")
    print_branch_list(agent.branch_list(), agent.current_branch)

    # Шаг 3: продолжить ветку «main» — риски и технические проблемы.
    console.rule("[bold]Шаг 3: ветка «main» — добавляем факты о рисках[/bold]", style="yellow")
    agent.branch_switch("main")
    main_answers: list[str] = []
    main_error: str | None = None

    for i, fact in enumerate(DEMO_BRANCH_MAIN_FACTS, start=1):
        console.print(f"[dim][main] факт {i}/{len(DEMO_BRANCH_MAIN_FACTS)} -> {fact}[/dim]")
        with console.status("[cyan]отправка...[/cyan]", spinner="dots"):
            try:
                _run_demo_request_with_retry(agent, fact)
            except Exception as e:
                main_error = str(e)
                print_error(f"Ошибка в ветке main: {main_error}")
                break

    if not main_error:
        for i, question in enumerate(DEMO_BRANCH_QUESTIONS, start=1):
            console.print(f"[dim][main] вопрос {i}/{len(DEMO_BRANCH_QUESTIONS)} -> {question}[/dim]")
            with console.status("[cyan]жду ответ...[/cyan]", spinner="dots"):
                try:
                    answer, _ = _run_demo_request_with_retry(agent, question)
                    main_answers.append(answer)
                except Exception as e:
                    main_error = str(e)
                    print_error(f"Ошибка в ветке main (вопросы): {main_error}")
                    break

    agent.branch_save("main")  # обновить снимок после продолжения

    # Шаг 4: переключиться на «alt» и добавить факты о фичах.
    console.rule("[bold]Шаг 4: ветка «alt» — добавляем факты о фичах[/bold]", style="magenta")
    agent.branch_switch("alt")
    alt_answers: list[str] = []
    alt_error: str | None = None

    for i, fact in enumerate(DEMO_BRANCH_ALT_FACTS, start=1):
        console.print(f"[dim][alt] факт {i}/{len(DEMO_BRANCH_ALT_FACTS)} -> {fact}[/dim]")
        with console.status("[cyan]отправка...[/cyan]", spinner="dots"):
            try:
                _run_demo_request_with_retry(agent, fact)
            except Exception as e:
                alt_error = str(e)
                print_error(f"Ошибка в ветке alt: {alt_error}")
                break

    if not alt_error:
        for i, question in enumerate(DEMO_BRANCH_QUESTIONS, start=1):
            console.print(f"[dim][alt] вопрос {i}/{len(DEMO_BRANCH_QUESTIONS)} -> {question}[/dim]")
            with console.status("[cyan]жду ответ...[/cyan]", spinner="dots"):
                try:
                    answer, _ = _run_demo_request_with_retry(agent, question)
                    alt_answers.append(answer)
                except Exception as e:
                    alt_error = str(e)
                    print_error(f"Ошибка в ветке alt (вопросы): {alt_error}")
                    break

    agent.branch_save("alt")

    # Шаг 5: итоговое сравнение.
    _print_demo_branch_results(
        main_answers=main_answers,
        alt_answers=alt_answers,
        main_error=main_error,
        alt_error=alt_error,
        agent=agent,
    )

    # Восстановить исходное состояние агента.
    agent.strategy = original_strategy
    if original_branch and original_branch in {b.name for b in agent.branch_list()}:
        try:
            agent.branch_switch(original_branch)
        except ValueError:
            pass


def _print_demo_branch_results(
    main_answers: list[str],
    alt_answers: list[str],
    main_error: str | None,
    alt_error: str | None,
    agent: Agent,
) -> None:
    console.print()
    console.rule("[bold cyan]DEMO BRANCH: сравнение веток «main» и «alt»[/bold cyan]")
    console.print()

    # Список веток.
    print_branch_list(agent.branch_list(), agent.current_branch)

    # Ответы по каждому вопросу бок о бок.
    for i, question in enumerate(DEMO_BRANCH_QUESTIONS, start=1):
        console.rule(f"[bold]Вопрос {i}[/bold]", style="dim")
        console.print(f"[bold]Вопрос:[/bold] {question}")

        main_snippet = (
            _one_line(main_answers[i - 1])[:300] if i <= len(main_answers) else
            f"(нет ответа{'— ошибка: ' + main_error if main_error else ''})"
        )
        alt_snippet = (
            _one_line(alt_answers[i - 1])[:300] if i <= len(alt_answers) else
            f"(нет ответа{'— ошибка: ' + alt_error if alt_error else ''})"
        )

        console.print(f"[bold yellow]main[/bold yellow]: {main_snippet}")
        console.print(f"[bold magenta]alt[/bold magenta]:  {alt_snippet}")
        console.print()

    console.print(
        "[dim]Ветки независимы: main помнит контекст рисков, alt — контекст фич. "
        "Переключайтесь командой /branch switch main|alt[/dim]"
    )
    console.print()


def _run_demo_memory(agent: Agent) -> None:
    """Демо модели памяти: 3 слоя, явное сохранение, влияние на ответы."""
    console.print()
    console.print(
        "[bold cyan]Demo-Memory запущено.[/bold cyan] "
        "Демонстрация 3 слоёв памяти: краткосрочной, рабочей и долговременной."
    )
    console.print(
        "[dim]Будет выполнено несколько API-запросов. Нажмите Ctrl+C для отмены.[/dim]\n"
    )

    original_strategy = agent.strategy
    original_task = agent.memory.working.task

    # ── Шаг 1: Очистка и исходное состояние ──────────────────────────────────
    console.rule("[bold]Шаг 1: Исходное состояние памяти (всё пусто)[/bold]", style="dim")
    agent.clear_history()
    agent.reset_session_metrics()
    agent.strategy = StrategyType.SUMMARY
    console.print("[dim]→ clear_history() выполнен, рабочая память очищена.[/dim]")
    print_memory_state(agent.memory, agent.history)

    # ── Шаг 2: Заполнить долговременную память (явно) ────────────────────────
    console.rule(
        "[bold magenta]Шаг 2: Заполняем долговременную память явно (/remember)[/bold magenta]"
    )
    console.print("[dim]Долговременная память — это то, что нужно помнить ВСЕГДА,\nдаже после /clear и перезапуска программы.[/dim]\n")

    console.print("[dim]→ /remember profile язык=Русский[/dim]")
    agent.memory.remember_profile("язык", "Русский")
    console.print("[dim]→ /remember profile роль=разработчик[/dim]")
    agent.memory.remember_profile("роль", "разработчик")
    console.print("[dim]→ /remember стиль=краткие ответы, без воды[/dim]")
    agent.memory.remember_knowledge("стиль", "краткие ответы, без воды")
    console.print("[dim]→ /remember decision Используем OpenRouter API для работы с LLM[/dim]")
    agent.memory.remember_decision("Используем OpenRouter API для работы с LLM")
    console.print()
    console.print("[green]✓ Долговременная память заполнена.[/green]\n")

    # ── Шаг 3: Задать задачу в рабочей памяти (явно) ─────────────────────────
    console.rule(
        "[bold cyan]Шаг 3: Задаём задачу в рабочей памяти (/task)[/bold cyan]"
    )
    console.print("[dim]Рабочая память — данные текущей задачи, очищается при /clear.[/dim]\n")

    task_text = "Разработать модуль памяти для LLM-ассистента"
    console.print(f"[dim]→ /task {task_text}[/dim]")
    agent.memory.set_task(task_text)
    agent.memory.add_goal("Реализовать 3 независимых слоя памяти")
    agent.memory.add_goal("Не сломать существующие 4 стратегии контекста")
    agent._save_state()
    console.print("[green]✓ Задача и цели установлены в рабочую память.[/green]\n")

    # ── Шаг 4: Отправить сообщение — авто-извлечение в рабочую память ────────
    console.rule(
        "[bold cyan]Шаг 4: Авто-извлечение фактов в рабочую память[/bold cyan]"
    )
    console.print("[dim]Пользователь называет детали проекта — они попадают в рабочую память автоматически.[/dim]\n")

    auto_facts_messages = [
        "проект: LLM Memory Demo",
        "стек: Python, Rich, Pydantic, OpenRouter API",
        "дедлайн: конец марта 2026",
    ]
    for msg in auto_facts_messages:
        console.print(f"[dim]→ отправляем: \"{msg}\"[/dim]")
        with console.status("[cyan]авто-извлечение...[/cyan]", spinner="dots"):
            try:
                _run_demo_request_with_retry(agent, msg)
            except Exception as e:
                print_error(f"Ошибка: {e}")
                agent.strategy = original_strategy
                return

    console.print("[green]✓ Факты авто-извлечены в рабочую память.[/green]\n")

    # ── Шаг 5: Показать состояние всех 3 слоёв памяти ────────────────────────
    console.rule("[bold]Шаг 5: Состояние всех 3 слоёв памяти[/bold]", style="yellow")
    console.print(
        "[dim]Теперь посмотрим что попало в каждый слой:[/dim]\n"
    )
    print_memory_state(agent.memory, agent.history, keep_last_n=4)

    # ── Шаг 6: Вопрос БЕЗ знания о памяти — чтобы показать влияние ───────────
    console.rule(
        "[bold yellow]Шаг 6: Контрольный вопрос — влияние памяти на ответ[/bold yellow]"
    )
    console.print(
        "[dim]Спрашиваем у ассистента о текущем контексте.\n"
        "В запрос автоматически вставлен блок [КОНТЕКСТ ПАМЯТИ], который содержит\n"
        "данные из рабочей и долговременной памяти.[/dim]\n"
    )

    question_with_memory = (
        "Коротко: какую задачу я сейчас решаю, что уже известно о проекте "
        "и в каком стиле тебе отвечать? Ответь в 3 пунктах."
    )
    console.print(f"[bold]Вопрос:[/bold] {question_with_memory}\n")
    with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
        try:
            answer_with_memory, _ = _run_demo_request_with_retry(agent, question_with_memory)
        except Exception as e:
            print_error(f"Ошибка: {e}")
            agent.strategy = original_strategy
            return
    console.print("[bold green]Ответ (память активна):[/bold green]")
    from .display import print_llm_response
    print_llm_response(answer_with_memory)

    # ── Шаг 7: Очищаем историю и проверяем что долговременная выжила ──────────
    console.rule(
        "[bold magenta]Шаг 7: /clear — рабочая память очищена, долговременная выжила[/bold magenta]"
    )
    console.print("[dim]→ выполняем /clear...[/dim]")
    agent.clear_history()
    console.print("[green]✓ История очищена. Краткосрочная и рабочая память — пусты.[/green]")
    console.print("[green]  Долговременная память сохранена (отдельный файл).[/green]\n")
    print_memory_state(agent.memory, agent.history)

    # ── Шаг 8: Тот же вопрос после /clear — долговременная всё ещё влияет ────
    console.rule(
        "[bold yellow]Шаг 8: Тот же вопрос после /clear — только долговременная память[/bold yellow]"
    )
    console.print(
        "[dim]Рабочей памяти больше нет, но профиль и знания из долговременной — остались.[/dim]\n"
    )
    console.print(f"[bold]Вопрос:[/bold] {question_with_memory}\n")
    with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
        try:
            answer_after_clear, _ = _run_demo_request_with_retry(agent, question_with_memory)
        except Exception as e:
            print_error(f"Ошибка: {e}")
            agent.strategy = original_strategy
            return
    console.print("[bold green]Ответ (только долговременная память):[/bold green]")
    print_llm_response(answer_after_clear)

    # ── Итог ──────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]ИТОГ: Demo-Memory[/bold cyan]")
    _print_demo_memory_summary(answer_with_memory, answer_after_clear)

    agent.strategy = original_strategy


def _print_demo_memory_summary(answer_with: str, answer_after: str) -> None:
    """Вывести итоговое сравнение ответов с памятью и после /clear."""
    table = Table(title="Влияние слоёв памяти на ответ", show_lines=True)
    table.add_column("Слой памяти", style="bold", min_width=22)
    table.add_column("Состояние", justify="center")
    table.add_column("Хранилище")
    table.add_column("Очищается при /clear", justify="center")

    table.add_row(
        "[yellow]Краткосрочная[/yellow]",
        "автоматически",
        "RAM (raw_history)",
        "[red]да[/red]",
    )
    table.add_row(
        "[cyan]Рабочая[/cyan]",
        "/task, авто-извлечение",
        "history.json",
        "[red]да[/red]",
    )
    table.add_row(
        "[magenta]Долговременная[/magenta]",
        "/remember (явно)",
        "long_term_memory.json",
        "[green]нет[/green]",
    )
    console.print()
    console.print(table)

    console.print()
    console.rule("[bold]Сравнение ответов[/bold]", style="dim")
    console.print(
        "[bold]До /clear[/bold] [dim](рабочая + долговременная):[/dim]\n"
        + _one_line(answer_with)[:300]
    )
    console.print()
    console.print(
        "[bold]После /clear[/bold] [dim](только долговременная):[/dim]\n"
        + _one_line(answer_after)[:300]
    )
    console.print()
    console.print(
        "[dim]Вывод: разные слои памяти хранятся независимо. "
        "Долговременная память пережила /clear и продолжает влиять на ответы. "
        "Используйте /memory, /task, /remember, /forget для явного управления.[/dim]"
    )
    console.print()


def _handle_profile(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/profile").strip()
    parts = raw.split(maxsplit=2)
    sub = parts[0].lower() if parts else ""

    mem = agent.memory

    # /profile  — показать активный профиль
    if not sub:
        profile = mem.get_active_profile()
        active_name = mem.get_active_profile_name()
        if profile is None:
            console.print("[dim]Активный профиль не выбран. Используйте /profile new <имя>.[/dim]\n")
        else:
            print_profile(profile, active_name, active=True)
        return

    # /profile list
    if sub == "list":
        profiles = mem.long_term.profiles
        active_name = mem.get_active_profile_name()
        print_profile_list(profiles, active_name)
        return

    # /profile new <имя>
    if sub == "new":
        name = parts[1].strip() if len(parts) > 1 else ""
        if not name:
            print_error("Использование: /profile new <имя>")
            return
        profile = _profile_wizard(name)
        mem.save_profile(name, profile)
        mem.switch_profile(name)
        console.print(f"[green]Профиль «{name}» создан и активирован.[/green]\n")
        print_profile(profile, name, active=True)
        return

    # /profile switch <имя>
    if sub == "switch":
        name = parts[1].strip() if len(parts) > 1 else ""
        if not name:
            print_error("Использование: /profile switch <имя>")
            return
        try:
            mem.switch_profile(name)
            profile = mem.get_active_profile()
            console.print(f"[green]Переключено на профиль «{name}».[/green]\n")
            if profile:
                print_profile(profile, name, active=True)
        except ValueError as e:
            print_error(str(e))
        return

    # /profile set <поле> <значение>
    if sub == "set":
        active_name = mem.get_active_profile_name()
        if not active_name:
            print_error("Нет активного профиля. Создайте: /profile new <имя>")
            return
        if len(parts) < 3:
            print_error(
                "Использование: /profile set <поле> <значение>\n"
                f"Поля: name, language, style {PROFILE_STYLE_OPTIONS}, "
                f"format {PROFILE_FORMAT_OPTIONS}, expertise {PROFILE_EXPERTISE_OPTIONS}, "
                "domain, constraint"
            )
            return
        field = parts[1].strip()
        value = parts[2].strip()
        try:
            known = mem.set_profile_field(active_name, field, value)
            if known:
                console.print(f"[green]Профиль «{active_name}»: {field} = {value}[/green]\n")
            else:
                print_error(
                    f"Неизвестное поле «{field}». "
                    "Доступные: name, language, style, format, expertise, domain, constraint"
                )
        except ValueError as e:
            print_error(str(e))
        return

    # /profile delete <имя>
    if sub == "delete":
        name = parts[1].strip() if len(parts) > 1 else ""
        if not name:
            print_error("Использование: /profile delete <имя>")
            return
        removed = mem.delete_profile(name)
        if removed:
            console.print(f"[green]Профиль «{name}» удалён.[/green]\n")
        else:
            print_error(f"Профиль «{name}» не найден.")
        return

    # /profile reset
    if sub == "reset":
        mem.deactivate_profile()
        console.print("[yellow]Активный профиль отключён. Используется дефолтное поведение.[/yellow]\n")
        return

    # /profile clear
    if sub == "clear":
        names = mem.list_profiles()
        if not names:
            console.print("[dim]Профилей нет.[/dim]\n")
            return
        for name in names:
            mem.delete_profile(name)
        console.print(f"[green]Удалено профилей: {len(names)}.[/green]\n")
        return

    print_error(
        f"Неизвестная подкоманда: «{sub}».\n"
        "Доступные: list, new, switch, set, delete, reset, clear"
    )


def _profile_wizard(name: str) -> UserProfile:
    """Интерактивный мастер создания профиля."""
    console.print(f"\n[bold cyan]Создание профиля «{name}»[/bold cyan]")
    console.print("[dim]Нажмите Enter чтобы оставить дефолтное значение.[/dim]\n")

    def ask(prompt: str, default: str, options: tuple[str, ...] | None = None) -> str:
        hint = f" [{default}]"
        if options:
            hint += f" ({'/'.join(options)})"
        console.print(f"  {prompt}{hint}: ", end="")
        try:
            val = input().strip()
        except EOFError:
            val = ""
        return val if val else default

    profile_name = ask("Ваше имя (для обращения)", "")
    language = ask("Язык ответов", "русский")
    style = ask("Стиль", "нейтральный", PROFILE_STYLE_OPTIONS)
    fmt = ask("Формат", "markdown", PROFILE_FORMAT_OPTIONS)
    expertise = ask("Уровень", "средний", PROFILE_EXPERTISE_OPTIONS)
    domain = ask("Область работы", "")

    constraints: list[str] = []
    console.print("  Ограничения (Enter для пропуска, пустая строка — конец):")
    for _ in range(5):
        console.print("    > ", end="")
        try:
            c = input().strip()
        except EOFError:
            break
        if not c:
            break
        constraints.append(c)

    console.print()
    return UserProfile(
        name=profile_name,
        language=language,
        style=style,
        format=fmt,
        expertise=expertise,
        domain=domain,
        constraints=constraints,
    )


# Вопрос для сравнения профилей в демо
DEMO_PERSONA_QUESTION = (
    "Объясни что такое Git rebase и чем он отличается от merge. "
    "Когда лучше использовать каждый из них?"
)

# Предустановленные профили для демо
DEMO_PROFILE_DEVELOPER = UserProfile(
    name="Алексей",
    language="русский",
    style="краткий",
    format="markdown",
    expertise="эксперт",
    domain="backend-разработка",
    constraints=["не объясняй базовые концепции Git", "только практические советы"],
)

DEMO_PROFILE_STUDENT = UserProfile(
    name="Маша",
    language="русский",
    style="подробный",
    format="plain",
    expertise="начинающий",
    domain="изучение программирования",
    constraints=["используй простые аналогии из реальной жизни", "избегай технического жаргона без объяснений"],
)


def _run_demo_persona(agent: Agent) -> None:
    """Демо персонализации: два профиля, один вопрос, сравнение ответов."""
    console.print()
    console.print(
        "[bold cyan]Demo-Persona запущено.[/bold cyan] "
        "Демонстрация влияния профиля пользователя на ответы ассистента."
    )
    console.print("[dim]Будет выполнено 4 API-запроса: профиль × вопрос + живое переключение.[/dim]\n")

    original_profile_name = agent.memory.get_active_profile_name()
    results: list[dict[str, object]] = []

    demo_profiles = [
        ("developer", DEMO_PROFILE_DEVELOPER),
        ("student", DEMO_PROFILE_STUDENT),
    ]

    for profile_name, profile in demo_profiles:
        console.rule(
            f"[bold]Профиль: «{profile_name}»[/bold]",
            style="cyan" if profile_name == "developer" else "magenta",
        )
        console.print("[dim]Параметры профиля:[/dim]")
        print_profile(profile, profile_name, active=True)

        console.print("[dim]Системный промпт, который увидит модель:[/dim]")
        console.print(
            Panel(
                profile.build_system_prompt(),
                border_style="dim",
                padding=(0, 1),
            )
        )

        agent.memory.save_profile(profile_name, profile)
        agent.memory.switch_profile(profile_name)
        agent.clear_history()

        console.print(f"\n[bold]Вопрос:[/bold] {DEMO_PERSONA_QUESTION}\n")
        with console.status(
            f"[bold cyan]«{profile_name}»: думаю...[/bold cyan]", spinner="dots"
        ):
            try:
                answer, _ = _run_demo_request_with_retry(agent, DEMO_PERSONA_QUESTION)
            except Exception as e:
                print_error(f"Ошибка: {e}")
                _restore_profile(agent, original_profile_name)
                return

        results.append({"profile": profile_name, "answer": answer})
        console.print(
            f"[bold {'cyan' if profile_name == 'developer' else 'magenta'}]"
            f"Ответ для «{profile_name}»:[/bold {'cyan' if profile_name == 'developer' else 'magenta'}]"
        )
        print_llm_response(answer)

    # ── Шаг 3: Живое переключение профилей ───────────────────────────────────
    console.rule(
        "[bold yellow]Живое переключение профилей[/bold yellow]"
    )
    console.print(
        "[dim]Теперь покажем переключение в реальном времени: один короткий вопрос,\n"
        "две смены профиля — ответы меняются автоматически.[/dim]\n"
    )

    switch_question = "Что такое Docker и зачем он нужен? Ответь в 2-3 предложениях."
    switch_profiles = [
        ("developer", DEMO_PROFILE_DEVELOPER, "cyan"),
        ("student", DEMO_PROFILE_STUDENT, "magenta"),
    ]

    for profile_name, profile, color in switch_profiles:
        agent.memory.save_profile(profile_name, profile)
        agent.memory.switch_profile(profile_name)
        agent.clear_history()

        console.print(
            f"[bold {color}]→ /profile switch {profile_name}[/bold {color}] "
            f"[dim](стиль={profile.style}, уровень={profile.expertise})[/dim]"
        )
        console.print(f"[bold]Вопрос:[/bold] {switch_question}")

        with console.status(
            f"[bold {color}]«{profile_name}»: думаю...[/bold {color}]", spinner="dots"
        ):
            try:
                switch_answer, _ = _run_demo_request_with_retry(agent, switch_question)
            except Exception as e:
                print_error(f"Ошибка: {e}")
                _restore_profile(agent, original_profile_name)
                return

        console.print(f"[bold {color}]Ответ «{profile_name}»:[/bold {color}]")
        print_llm_response(switch_answer)

    _restore_profile(agent, original_profile_name)
    _print_demo_persona_results(results)


def _restore_profile(agent: Agent, original_name: str) -> None:
    if original_name and original_name in agent.memory.list_profiles():
        try:
            agent.memory.switch_profile(original_name)
        except ValueError:
            agent.memory.deactivate_profile()
    else:
        agent.memory.deactivate_profile()
    agent.clear_history()


def _print_demo_persona_results(results: list[dict[str, object]]) -> None:
    console.rule("[bold cyan]ИТОГ: Demo-Persona[/bold cyan]")
    console.print()

    table = Table(title="Сравнение профилей", show_lines=True)
    table.add_column("Параметр", style="bold", min_width=16)
    table.add_column("developer (эксперт)", style="cyan")
    table.add_column("student (начинающий)", style="magenta")

    table.add_row("Стиль", "краткий", "подробный")
    table.add_row("Формат", "markdown", "plain")
    table.add_row("Уровень", "эксперт", "начинающий")
    table.add_row("Ограничения", "без основ Git", "аналогии, без жаргона")
    console.print(table)
    console.print()

    if len(results) == 2:
        dev_ans = _one_line(str(results[0]["answer"]))[:300]
        stu_ans = _one_line(str(results[1]["answer"]))[:300]
        console.rule("[bold]Краткое сравнение ответов[/bold]", style="dim")
        console.print(f"[bold cyan]developer:[/bold cyan] {dev_ans}")
        console.print()
        console.print(f"[bold magenta]student:[/bold magenta] {stu_ans}")
        console.print()

    console.print(
        "[dim]Вывод: один и тот же вопрос — принципиально разные ответы.\n"
        "Профиль автоматически меняет стиль, глубину и формат без явных инструкций в вопросе.\n"
        "Используйте /profile new <имя> чтобы создать свой профиль.[/dim]"
    )
    console.print()


def _handle_task(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/task").strip()
    if not raw:
        wm = agent.memory.working
        if wm.task_fsm is not None:
            print_task_fsm(wm.task_fsm)
        elif wm.task:
            console.print(f"[dim]Текущая задача: {wm.task}[/dim]\n")
        else:
            console.print(
                "[dim]Задача не задана.\n"
                "  /task <описание>             — задать текст задачи\n"
                "  /task fsm start <имя>        — запустить Task FSM[/dim]\n"
            )
        return
    if raw.lower() == "clear":
        agent.memory.clear_working()
        agent._save_state()
        console.print("[green]Рабочая память очищена.[/green]\n")
        return

    # ── FSM-подкоманды (/task fsm ...) ───────────────────────────────────────
    if raw.lower().startswith("fsm"):
        _handle_task_fsm(raw[3:].strip(), agent)
        return

    agent.memory.set_task(raw)
    agent._save_state()
    console.print(f"[green]Задача установлена:[/green] {raw}\n")


def _handle_task_fsm(raw: str, agent: Agent) -> None:
    """Обработчик подкоманд /task fsm ..."""
    parts = raw.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    mem = agent.memory

    # /task fsm  (без подкоманды) → показать статус
    if not sub or sub == "status":
        fsm = mem.get_fsm()
        if fsm is None:
            console.print(
                "[dim]FSM не активен. Запустите: /task fsm start <имя задачи>[/dim]\n"
            )
        else:
            print_task_fsm(fsm)
        return

    # /task fsm start <имя>
    if sub == "start":
        if not rest:
            print_error("Использование: /task fsm start <имя задачи>")
            return
        fsm = mem.start_fsm(rest)
        agent._save_state()
        console.print(f"[green]FSM запущен:[/green] «{fsm.task_name}» → этап [yellow]Планирование[/yellow]\n")
        print_task_fsm(fsm)
        return

    # /task fsm next [заметка]
    if sub == "next":
        try:
            fsm = mem.advance_fsm(note=rest)
            agent._save_state()
            from .task_fsm import STAGE_LABELS as FSM_STAGE_LABELS  # noqa: PLC0415
            label = FSM_STAGE_LABELS[fsm.stage]
            console.print(f"[green]Переход выполнен → этап [bold]{label}[/bold][/green]\n")
            print_task_fsm(fsm)
        except ValueError as e:
            print_error(str(e))
        return

    # /task fsm pause
    if sub == "pause":
        try:
            fsm = mem.pause_fsm()
            agent._save_state()
            console.print("[yellow]FSM поставлен на паузу.[/yellow] Агент отвечает свободно.\n")
            print_task_fsm(fsm)
        except ValueError as e:
            print_error(str(e))
        return

    # /task fsm resume
    if sub == "resume":
        try:
            fsm = mem.resume_fsm()
            agent._save_state()
            from .task_fsm import STAGE_LABELS as FSM_STAGE_LABELS  # noqa: PLC0415
            label = FSM_STAGE_LABELS[fsm.stage]
            console.print(f"[green]FSM возобновлён → этап [bold]{label}[/bold][/green]\n")
            print_task_fsm(fsm)
        except ValueError as e:
            print_error(str(e))
        return

    # /task fsm step <описание>
    if sub == "step":
        if not rest:
            print_error("Использование: /task fsm step <описание текущего шага>")
            return
        try:
            mem.set_fsm_step(rest)
            agent._save_state()
            console.print(f"[green]Шаг установлен:[/green] {rest}\n")
        except ValueError as e:
            print_error(str(e))
        return

    # /task fsm artifact <ключ> <текст>
    if sub == "artifact":
        art_parts = rest.split(maxsplit=1)
        if len(art_parts) < 2:
            print_error("Использование: /task fsm artifact <ключ> <текст>")
            return
        try:
            fsm = mem.add_fsm_artifact(art_parts[0], art_parts[1])
            agent._save_state()
            console.print(
                f"[green]Артефакт сохранён:[/green] [cyan]{art_parts[0]}[/cyan] = {art_parts[1][:60]}\n"
            )
        except ValueError as e:
            print_error(str(e))
        return

    # /task fsm clear
    if sub == "clear":
        mem.clear_fsm()
        agent._save_state()
        console.print("[yellow]FSM сброшен.[/yellow]\n")
        return

    print_error(
        f"Неизвестная FSM-подкоманда: «{sub}».\n"
        "Доступные: start, status, next, pause, resume, step, artifact, clear"
    )


def _handle_remember(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/remember").strip()
    if not raw:
        console.print(
            "[dim]Использование:\n"
            "  /remember <текст>                      — заметка\n"
            "  /remember <ключ>=<значение>             — знание\n"
            "  /remember profile <ключ>=<значение>     — профиль\n"
            "  /remember decision <текст>              — решение[/dim]\n"
        )
        return

    # /remember profile key=value
    if raw.lower().startswith("profile "):
        rest = raw[len("profile "):].strip()
        if "=" in rest:
            key, _, value = rest.partition("=")
            if key.strip() and value.strip():
                agent.memory.remember_profile(key.strip(), value.strip())
                console.print(
                    f"[green]Профиль обновлён:[/green] {key.strip()} = {value.strip()}\n"
                )
                return
        print_error("Формат: /remember profile <ключ>=<значение>")
        return

    # /remember decision <text>
    if raw.lower().startswith("decision "):
        decision_text = raw[len("decision "):].strip()
        if decision_text:
            agent.memory.remember_decision(decision_text)
            console.print(f"[green]Решение сохранено:[/green] {decision_text}\n")
            return
        print_error("Формат: /remember decision <текст>")
        return

    # /remember key=value  → knowledge
    if "=" in raw and not raw.startswith("="):
        key, _, value = raw.partition("=")
        if key.strip() and value.strip():
            agent.memory.remember_knowledge(key.strip(), value.strip())
            console.print(
                f"[green]Знание сохранено:[/green] {key.strip()} = {value.strip()}\n"
            )
            return

    # /remember <текст>  → note
    agent.memory.remember_note(raw)
    console.print(f"[green]Заметка сохранена:[/green] {raw}\n")


def _handle_forget(text: str, agent: Agent) -> None:
    raw = text.removeprefix("/forget").strip()
    if not raw:
        console.print("[dim]Использование: /forget <ключ>[/dim]\n")
        return
    removed = agent.memory.forget(raw)
    if removed:
        console.print(f"[green]Удалено из долговременной памяти:[/green] {raw}\n")
    else:
        console.print(
            f"[yellow]Ключ «{raw}» не найден в knowledge или profile долговременной памяти.[/yellow]\n"
        )


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


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Task FSM
# ─────────────────────────────────────────────────────────────────────────────

# Тема задачи для всего демо-сценария
_DEMO_FSM_TASK = "Реализовать REST-эндпоинт для экспорта отчёта в PDF"

# Один вопрос, который задаётся в каждом этапе — чтобы было чётко видно
# как ОДНА и та же фраза получает разный ответ в зависимости от stage-промпта
_DEMO_FSM_PROBE = (
    "Что мне сделать прямо сейчас по задаче «{}»? "
    "Дай конкретный следующий шаг в 2-3 предложениях."
)


def _run_demo_fsm(agent: Agent) -> None:
    """Демо Task FSM: planning→execution→validation→done, пауза/возобновление."""
    from .task_fsm import STAGE_SYSTEM_PROMPTS, TaskStage

    console.print()
    console.print(
        "[bold cyan]Demo-FSM запущено.[/bold cyan] "
        "Демонстрация детерминированного конечного автомата задачи."
    )
    console.print(
        "[dim]Жизненный цикл: planning → execution → validation → done\n"
        "Один и тот же вопрос — принципиально разные ответы в каждом этапе.\n"
        "Будет выполнено ~6 API-запросов. Нажмите Ctrl+C для отмены.[/dim]\n"
    )

    # Сохраняем исходное состояние, чтобы восстановить после демо
    original_fsm = agent.memory.get_fsm()
    original_task = agent.memory.working.task

    agent.clear_history()
    agent.memory.clear_fsm()
    agent._save_state()

    results: list[dict[str, object]] = []

    def _ask_probe(stage_label: str) -> str | None:
        question = _DEMO_FSM_PROBE.format(_DEMO_FSM_TASK)
        console.print(f"\n[bold]Вопрос:[/bold] {question}\n")
        with console.status(
            f"[bold cyan]{stage_label}: думаю...[/bold cyan]", spinner="dots"
        ):
            try:
                answer, _ = _run_demo_request_with_retry(agent, question)
                return answer
            except Exception as e:
                print_error(f"Ошибка API: {e}")
                return None

    # ── Шаг 1: Запускаем FSM ─────────────────────────────────────────────────
    console.rule("[bold yellow]Шаг 1: Запуск FSM (/task fsm start)[/bold yellow]")
    console.print(
        "[dim]Запускаем FSM. Этап PLANNING — агент помогает планировать,\n"
        "не начиная реализацию. Ниже показан stage-промпт, который\n"
        "будет автоматически добавлен к каждому запросу LLM:[/dim]\n"
    )
    fsm = agent.memory.start_fsm(_DEMO_FSM_TASK)
    agent._save_state()
    console.print(
        Panel(
            STAGE_SYSTEM_PROMPTS[TaskStage.PLANNING],
            title="[dim]system-промпт этапа PLANNING[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    print_task_fsm(fsm)

    # Вопрос на этапе PLANNING
    console.rule("[bold yellow]Этап PLANNING → вопрос агенту[/bold yellow]")
    console.print(
        "[dim]Смотрим как агент отвечает в режиме планирования.[/dim]"
    )
    fsm.set_step("Формулировка требований", "Определить входные параметры и формат вывода")
    agent._save_state()
    answer_planning = _ask_probe("planning")
    if answer_planning is None:
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    results.append({"stage": "planning", "answer": answer_planning})
    print_llm_response(answer_planning)

    # Сохраняем артефакт этапа
    agent.memory.add_fsm_artifact(
        "план",
        "Эндпоинт GET /reports/{id}/export, параметры: format=pdf, locale, date_range",
    )
    agent._save_state()
    console.print("[dim]→ /task fsm artifact план «Эндпоинт GET /reports/{id}/export…»[/dim]\n")

    # ── Шаг 2: Переход в EXECUTION ────────────────────────────────────────────
    console.rule("[bold cyan]Шаг 2: Переход в EXECUTION (/task fsm next)[/bold cyan]")
    console.print(
        "[dim]Переходим к реализации. Stage-промпт меняется:\n"
        "теперь агент фокусируется на конкретных действиях по плану.[/dim]\n"
    )
    try:
        fsm = agent.memory.advance_fsm("план утверждён")
    except ValueError as e:
        print_error(str(e))
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    agent._save_state()
    console.print(
        Panel(
            STAGE_SYSTEM_PROMPTS[TaskStage.EXECUTION],
            title="[dim]system-промпт этапа EXECUTION[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    print_task_fsm(fsm)

    console.rule("[bold cyan]Этап EXECUTION → тот же вопрос[/bold cyan]")
    console.print(
        "[dim]Тот же вопрос — но агент теперь в режиме реализации.[/dim]"
    )
    fsm.set_step("Написать обработчик эндпоинта", "Сгенерировать PDF через библиотеку")
    agent._save_state()
    answer_execution = _ask_probe("execution")
    if answer_execution is None:
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    results.append({"stage": "execution", "answer": answer_execution})
    print_llm_response(answer_execution)

    # ── Шаг 3: ПАУЗА ─────────────────────────────────────────────────────────
    console.rule("[bold magenta]Шаг 3: Пауза (/task fsm pause)[/bold magenta]")
    console.print(
        "[dim]Ставим задачу на паузу — пришёл срочный вопрос не по теме.\n"
        "Stage-промпт убирается: LLM отвечает свободно, без ограничений этапа.[/dim]\n"
    )
    try:
        fsm = agent.memory.pause_fsm()
    except ValueError as e:
        print_error(str(e))
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    agent._save_state()
    print_task_fsm(fsm)

    pause_question = "Что такое JWT токен и зачем он нужен? Ответь в 2 предложениях."
    console.print(f"[bold]Посторонний вопрос:[/bold] {pause_question}\n")
    with console.status("[bold magenta]пауза: думаю...[/bold magenta]", spinner="dots"):
        try:
            answer_pause, _ = _run_demo_request_with_retry(agent, pause_question)
        except Exception as e:
            print_error(f"Ошибка API: {e}")
            _restore_demo_fsm(agent, original_fsm, original_task)
            return
    console.print("[bold magenta]Ответ (пауза, агент свободен):[/bold magenta]")
    print_llm_response(answer_pause)
    console.print(
        "[dim]Агент ответил без каких-либо FSM-ограничений.\n"
        "Состояние задачи сохранено — продолжим с того же места.[/dim]\n"
    )

    # ── Шаг 4: ВОЗОБНОВЛЕНИЕ ─────────────────────────────────────────────────
    console.rule("[bold cyan]Шаг 4: Возобновление (/task fsm resume)[/bold cyan]")
    console.print(
        "[dim]Возобновляем задачу. Stage-промпт EXECUTION возвращается автоматически.\n"
        "Не нужно ничего объяснять заново — контекст сохранён в FSM.[/dim]\n"
    )
    try:
        fsm = agent.memory.resume_fsm()
    except ValueError as e:
        print_error(str(e))
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    agent._save_state()
    print_task_fsm(fsm)

    console.rule("[bold cyan]После resume → тот же вопрос снова[/bold cyan]")
    console.print(
        "[dim]Тот же вопрос сразу после resume: агент снова в режиме EXECUTION.[/dim]"
    )
    answer_resumed = _ask_probe("execution (после resume)")
    if answer_resumed is None:
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    results.append({"stage": "execution_resumed", "answer": answer_resumed})
    print_llm_response(answer_resumed)

    # ── Шаг 5: Переход в VALIDATION ──────────────────────────────────────────
    console.rule("[bold green]Шаг 5: Переход в VALIDATION (/task fsm next)[/bold green]")
    console.print(
        "[dim]Реализация готова. Переходим в режим валидации — агент должен\n"
        "критически проверить результат и дать вердикт.[/dim]\n"
    )
    try:
        fsm = agent.memory.advance_fsm("реализация завершена")
    except ValueError as e:
        print_error(str(e))
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    agent._save_state()
    console.print(
        Panel(
            STAGE_SYSTEM_PROMPTS[TaskStage.VALIDATION],
            title="[dim]system-промпт этапа VALIDATION[/dim]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    print_task_fsm(fsm)

    console.rule("[bold green]Этап VALIDATION → тот же вопрос[/bold green]")
    console.print("[dim]Тот же вопрос — агент в режиме проверки.[/dim]")
    fsm.set_step("Проверка артефактов", "Вынести вердикт: принять / доработать / отклонить")
    agent._save_state()
    answer_validation = _ask_probe("validation")
    if answer_validation is None:
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    results.append({"stage": "validation", "answer": answer_validation})
    print_llm_response(answer_validation)

    # Сохраняем артефакт валидации
    agent.memory.add_fsm_artifact("вердикт", "принять — базовая реализация готова к ревью")
    agent._save_state()
    console.print("[dim]→ /task fsm artifact вердикт «принять — базовая реализация…»[/dim]\n")

    # ── Шаг 6: DONE ───────────────────────────────────────────────────────────
    console.rule("[bold green]Шаг 6: Завершение (/task fsm next → done)[/bold green]")
    try:
        fsm = agent.memory.advance_fsm("валидация пройдена")
    except ValueError as e:
        print_error(str(e))
        _restore_demo_fsm(agent, original_fsm, original_task)
        return
    agent._save_state()
    print_task_fsm(fsm)
    console.print("[green]Задача завершена. Stage-промпт больше не инжектируется.[/green]\n")

    # ── Итоговая таблица ──────────────────────────────────────────────────────
    _print_demo_fsm_summary(results)

    _restore_demo_fsm(agent, original_fsm, original_task)


def _restore_demo_fsm(agent: Agent, original_fsm: object, original_task: str) -> None:
    """Восстановить состояние после демо."""
    from .task_fsm import TaskFSM
    agent.memory.clear_fsm()
    if original_fsm is not None and isinstance(original_fsm, TaskFSM):
        agent.memory.working.task_fsm = original_fsm
    if original_task:
        agent.memory.set_task(original_task)
    agent.clear_history()
    agent._save_state()


def _print_demo_fsm_summary(results: list[dict[str, object]]) -> None:
    """Итоговая таблица сравнения ответов по этапам."""
    console.rule("[bold cyan]ИТОГ: Demo-FSM[/bold cyan]")
    console.print()

    stage_colors = {
        "planning": "yellow",
        "execution": "cyan",
        "execution_resumed": "cyan",
        "validation": "green",
    }
    stage_labels = {
        "planning": "Планирование",
        "execution": "Реализация",
        "execution_resumed": "Реализация (после resume)",
        "validation": "Валидация",
    }

    table = Table(title="Один вопрос — разные этапы — разные ответы", show_lines=True)
    table.add_column("Этап", style="bold", min_width=24)
    table.add_column("Фокус агента")
    table.add_column("Краткий ответ", min_width=40)

    for r in results:
        stage = str(r["stage"])
        answer = _one_line(str(r["answer"]))[:200]
        color = stage_colors.get(stage, "white")
        label = stage_labels.get(stage, stage)
        focus_map = {
            "planning": "требования, шаги, риски",
            "execution": "конкретные действия, код",
            "execution_resumed": "продолжение реализации",
            "validation": "проверка, вердикт",
        }
        table.add_row(
            f"[{color}]{label}[/{color}]",
            focus_map.get(stage, "—"),
            answer,
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Вывод: stage-промпт меняет роль агента на каждом этапе без изменения вопроса.\n"
        "Переходы детерминированы — LLM не решает когда переключаться, это делает пользователь.\n"
        "Пауза/resume сохраняет этап и контекст — продолжение без повторных объяснений.\n\n"
        "Команды для ручного использования:\n"
        "  /task fsm start <имя>     — запустить\n"
        "  /task fsm next            — следующий этап\n"
        "  /task fsm pause / resume  — пауза и возобновление\n"
        "  /task fsm status          — текущее состояние[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Обработчики команд инвариантов
# ─────────────────────────────────────────────────────────────────────────────

def _handle_invariant_add(text: str, agent: Agent) -> None:
    """Парсит: /invariant-add <кат> <название> | <описание>"""
    from .invariants import CATEGORY_ALIASES, InvariantCategory

    raw = text.removeprefix("/invariant-add").strip()
    if not raw:
        console.print(
            "[dim]Использование: /invariant-add <кат> <название> | <описание>\n"
            "Категории: arch · tech · stack · biz[/dim]\n"
        )
        return

    parts = raw.split(None, 1)
    if len(parts) < 2:
        print_error("Укажите категорию и название: /invariant-add arch <название> | <описание>")
        return

    cat_raw, rest = parts[0].lower(), parts[1]
    cat_key = CATEGORY_ALIASES.get(cat_raw)
    if cat_key is None:
        print_error(
            f"Неизвестная категория: «{cat_raw}». "
            "Допустимые: arch, tech, stack, biz"
        )
        return

    category = InvariantCategory(cat_key)

    if "|" in rest:
        title_raw, description = rest.split("|", 1)
        title = title_raw.strip()
        description = description.strip()
    else:
        title = rest.strip()
        description = title

    if not title:
        print_error("Название инварианта не может быть пустым.")
        return

    inv = agent.invariants.add(category=category, title=title, description=description)
    console.print(
        f"[green]Инвариант добавлен:[/green] [{inv.id}] {inv.title}\n"
        f"[dim]Категория: {inv.category.value}  |  {inv.description}[/dim]\n"
    )


def _handle_invariant_del(text: str, agent: Agent) -> None:
    inv_id = text.removeprefix("/invariant-del").strip().upper()
    if not inv_id:
        print_error("Укажите ID: /invariant-del INV-XXXXXX")
        return
    if agent.invariants.remove(inv_id):
        console.print(f"[green]Инвариант {inv_id} удалён.[/green]\n")
    else:
        print_error(f"Инвариант с ID «{inv_id}» не найден.")


# ─────────────────────────────────────────────────────────────────────────────
# Демо инвариантов
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_INVARIANTS = [
    ("arch", "Только Clean Architecture",
     "Запрещены прямые обращения из UI-слоя в базу данных или бизнес-логику. "
     "Всегда используй слои: Presentation → Domain → Data."),
    ("stack", "Backend только на FastAPI + PostgreSQL",
     "Запрещено вводить любые другие фреймворки (Django, Flask) или СУБД (MongoDB, Redis как primary storage). "
     "Разрешённый стек: FastAPI, SQLAlchemy, Alembic, PostgreSQL ≤14."),
    ("stack", "Frontend только на React (TypeScript)",
     "Запрещено использовать Vue, Angular, Svelte или другие UI-фреймворки. "
     "Стейт-менеджмент: только Zustand или React Context."),
    ("biz", "Данные пользователей не удаляются физически",
     "Любые объекты пользователей (аккаунты, заявки, документы) ТОЛЬКО помечаются как archived=true. "
     "Физическое DELETE запрещено — необходимо для аудита и GDPR-compliant восстановления."),
    ("biz", "Все API-ответы содержат request_id",
     "Каждый HTTP-ответ обязан включать заголовок X-Request-ID для трассировки. "
     "Исключений нет — это требование безопасности и поддержки."),
]

_DEMO_CONFLICT_REQUESTS = [
    (
        "Напиши React-компонент, который напрямую делает SQL-запрос к PostgreSQL.",
        "Нарушает: Clean Architecture (UI → DB напрямую)",
    ),
    (
        "Предложи использовать MongoDB для хранения профилей пользователей вместо PostgreSQL.",
        "Нарушает: ограничение стека (только PostgreSQL)",
    ),
    (
        "Напиши endpoint DELETE /users/{id}, который физически удаляет пользователя из БД.",
        "Нарушает: бизнес-правило (no physical delete)",
    ),
]

_DEMO_VALID_REQUEST = (
    "Как правильно организовать слои Clean Architecture для модуля авторизации в FastAPI?"
)


def _run_demo_invariants(agent: Agent) -> None:
    """Демо инвариантов: добавление, корректный запрос, конфликтующий запрос."""
    console.rule("[bold magenta]ДЕМО: Инварианты ассистента[/bold magenta]")
    console.print()
    console.print(
        "[dim]Демонстрирует: хранение инвариантов отдельно от диалога, "
        "учёт в рассуждениях, отказ при конфликте с объяснением.[/dim]\n"
    )

    # 1. Сохраняем текущие инварианты и очищаем для чистоты демо
    original_invariants = list(agent.invariants.invariants)
    agent.invariants.clear()

    # 2. Загружаем демо-инварианты
    from .invariants import CATEGORY_ALIASES, InvariantCategory

    console.print("[bold]Шаг 1: Загружаем инварианты проекта[/bold]")
    added = []
    for cat_alias, title, description in _DEMO_INVARIANTS:
        cat = InvariantCategory(CATEGORY_ALIASES[cat_alias])
        inv = agent.invariants.add(category=cat, title=title, description=description)
        added.append(inv)
        console.print(f"  [green]+[/green] [{inv.id}] {inv.title}")

    console.print()
    print_invariants(agent.invariants)

    # 3. Корректный запрос (должен пройти нормально)
    console.rule("[bold cyan]Шаг 2: Корректный запрос (соответствует инвариантам)[/bold cyan]")
    console.print(f"[bold yellow]Запрос:[/bold yellow] {_DEMO_VALID_REQUEST}\n")
    with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
        try:
            reply, stats = agent.run_with_stats(_DEMO_VALID_REQUEST)
            from .display import print_llm_response, print_chat_turn_stats
            print_llm_response(reply)
            print_chat_turn_stats(stats)
        except Exception as e:
            print_error(str(e))

    # 4. Конфликтующие запросы
    for i, (request, expected_conflict) in enumerate(_DEMO_CONFLICT_REQUESTS, 1):
        console.rule(
            f"[bold red]Шаг {2 + i}: Конфликтующий запрос #{i}[/bold red]"
        )
        console.print(f"[bold yellow]Запрос:[/bold yellow] {request}")
        console.print(f"[dim]Ожидаемый конфликт: {expected_conflict}[/dim]\n")

        with console.status("[bold cyan]Думаю...[/bold cyan]", spinner="dots"):
            try:
                reply, stats = agent.run_with_stats(request)
                from .display import print_llm_response, print_chat_turn_stats
                print_llm_response(reply)
                print_chat_turn_stats(stats)
            except Exception as e:
                print_error(str(e))

    # 5. Восстанавливаем оригинальное состояние
    agent.invariants.clear()
    for inv in original_invariants:
        agent.invariants.add(
            category=inv.category, title=inv.title, description=inv.description
        )
    agent.clear_history()

    console.rule("[bold magenta]ДЕМО завершено[/bold magenta]")
    console.print()
    console.print(
        "[dim]Инварианты хранятся в ~/.config/llm-cli/invariants.json — отдельно от диалога.\n"
        "Они инжектируются в каждый запрос как системный блок с явными инструкциями:\n"
        "  • учитывать инварианты в рассуждениях\n"
        "  • отказывать при конфликте\n"
        "  • называть нарушенный инвариант по ID\n"
        "  • предлагать допустимую альтернативу\n\n"
        "Команды: /invariants · /invariant-add · /invariant-del · /invariant-clear[/dim]"
    )
    console.print()
