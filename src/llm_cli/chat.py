"""Интерактивный чат-цикл с LLM."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from .agent import Agent
from .api import OpenRouterClient
from .benchmark import run_benchmark
from .config import AppConfig
from .display import print_chat_turn_stats, print_error, print_llm_response, print_welcome
from .models import BENCHMARK_PROMPT

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
    _print_compression_status(agent)

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

        if text == "/compare":
            console.print()
            run_benchmark(client, benchmark_prompt, cfg.models, agent.temperature)
            continue

        if text == "/demo-compare":
            _run_demo_compare(agent)
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
        _print_compression_status(agent)
        return

    if raw in ("on", "1", "true", "yes"):
        agent.compression_enabled = True
        console.print("[green]Компрессия контекста: ON[/green]\n")
        _print_compression_status(agent)
        return

    if raw in ("off", "0", "false", "no"):
        agent.compression_enabled = False
        console.print("[yellow]Компрессия контекста: OFF[/yellow]\n")
        _print_compression_status(agent)
        return

    print_error("Использование: /compress on|off")


def _print_compression_status(agent: Agent) -> None:
    status = agent.compression_status
    mode = "ON" if status.enabled else "OFF"
    console.print(
        "[dim]"
        f"Компрессия: {mode} | keep_last_n={status.keep_last_n} | "
        f"summarize_every={status.summarize_every} | "
        f"min_messages_for_summary={status.min_messages_for_summary} | "
        f"summary_chars={status.summary_chars} | "
        f"сжато сообщений={status.compressed_messages_count}"
        "[/dim]"
    )
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
