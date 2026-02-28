"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import (
    print_branch_list,
    print_chat_turn_stats,
    print_error,
    print_llm_response,
    print_strategy_status,
    print_welcome,
)
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
