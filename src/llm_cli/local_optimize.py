"""Демо-режим оптимизации локальной LLM: сравнение параметров и prompt-шаблонов."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .api import OpenRouterClient
from .models import ChatMessage

if TYPE_CHECKING:
    pass

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Тестовый вопрос — одинаковый для всех конфигураций
# ─────────────────────────────────────────────────────────────────────────────

_TEST_QUESTION = (
    "Объясни, как работает механизм Attention в трансформерах. "
    "Почему он лучше RNN для длинных последовательностей? "
    "Ответь чётко и структурированно."
)

# ─────────────────────────────────────────────────────────────────────────────
# Этап 1: конфигурации параметров
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamConfig:
    label: str
    tier: str
    temperature: float
    max_tokens: int
    description: str


PARAM_CONFIGS: list[ParamConfig] = [
    ParamConfig(
        label="До оптимизации",
        tier="baseline",
        temperature=0.8,
        max_tokens=512,
        description="Высокая температура — непредсказуемые ответы, избыточный вывод",
    ),
    ParamConfig(
        label="Оптимальный",
        tier="optimal",
        temperature=0.2,
        max_tokens=256,
        description="Низкая температура — точные ответы, ограниченный вывод",
    ),
    ParamConfig(
        label="Креативный",
        tier="creative",
        temperature=1.2,
        max_tokens=512,
        description="Очень высокая температура — нестандартные формулировки",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Этап 2: prompt-шаблоны
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptConfig:
    label: str
    system_prompt: str
    description: str


PROMPT_CONFIGS: list[PromptConfig] = [
    PromptConfig(
        label="Базовый промпт",
        system_prompt="Ты полезный ассистент.",
        description="Общий промпт без специализации",
    ),
    PromptConfig(
        label="Специализированный",
        system_prompt=(
            "Ты эксперт по глубокому обучению и NLP. "
            "Объясняй технические концепции чётко, используй термины корректно. "
            "Структурируй ответ: сначала суть, затем детали. "
            "Технические термины (Attention, RNN, Transformer, softmax и т.п.) пиши на английском. "
            "Объяснения — на русском языке. Будь краток и точен."
        ),
        description="Роль + ограничения + формат + язык",
    ),
    PromptConfig(
        label="RAG-стиль",
        system_prompt=(
            "Ты технический ассистент. Отвечай строго по следующему шаблону:\n"
            "1. СУТЬ (1-2 предложения)\n"
            "2. КАК РАБОТАЕТ (3-4 пункта)\n"
            "3. ПРЕИМУЩЕСТВО (1-2 предложения)\n\n"
            "Технические термины — английский. Объяснения — русский. Без лишних вступлений."
        ),
        description="Жёсткий шаблон ответа с нумерованными секциями",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Результат одного запроса
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    label: str
    description: str
    answer: str
    duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float:
        if self.duration_ms == 0:
            return 0.0
        return self.completion_tokens / (self.duration_ms / 1000)

    @property
    def answer_words(self) -> int:
        return len(self.answer.split())


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты вывода
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(base_url: str, model_id: str) -> None:
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("Оптимизация локальной LLM\n", "bold white"),
            ("Qwen3.5-27B · Q4_K_M · llama.cpp", "dim cyan"),
        ),
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()
    console.print(f"  [dim]Сервер:[/dim] [cyan]{base_url}[/cyan]")
    console.print(f"  [dim]Модель:[/dim] [cyan]{model_id}[/cyan]")
    console.print(f"  [dim]Вопрос:[/dim] {_TEST_QUESTION[:80]}...")
    console.print()


def _run_single(
    client: OpenRouterClient,
    model_id: str,
    system_prompt: str,
    question: str,
    temperature: float,
    max_tokens: int,
    label: str,
    description: str,
) -> RunResult:
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=question),
    ]
    try:
        start = time.perf_counter_ns()
        response = client.send_raw(
            messages,
            model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration_ms = (time.perf_counter_ns() - start) // 1_000_000

        if response.error:
            return RunResult(
                label=label,
                description=description,
                answer="",
                duration_ms=duration_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error=response.error.message,
            )

        content = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage
        return RunResult(
            label=label,
            description=description,
            answer=content,
            duration_ms=duration_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
    except Exception as exc:
        return RunResult(
            label=label,
            description=description,
            answer="",
            duration_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            error=str(exc),
        )


def _print_answer_preview(result: RunResult, max_lines: int = 6) -> None:
    if result.error:
        console.print(f"  [red]Ошибка:[/red] {result.error}")
        return
    lines = result.answer.strip().splitlines()
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n[dim]... ({len(lines) - max_lines} строк скрыто)[/dim]"
    console.print(Panel(
        preview,
        title=f"[bold]{result.label}[/bold]",
        border_style="dim",
        padding=(0, 1),
    ))


def _print_metrics_table(results: list[RunResult], title: str) -> None:
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("Конфигурация", style="bold white", min_width=22)
    table.add_column("Время (мс)", justify="right", style="yellow")
    table.add_column("tok/sec", justify="right", style="green")
    table.add_column("Prompt tok", justify="right", style="dim")
    table.add_column("Output tok", justify="right", style="cyan")
    table.add_column("Слов", justify="right", style="dim")
    table.add_column("Статус", justify="center")

    for r in results:
        status = "[red]ОШИБКА[/red]" if r.error else "[green]OK[/green]"
        table.add_row(
            r.label,
            f"{r.duration_ms:,}",
            f"{r.tokens_per_sec:.1f}",
            str(r.prompt_tokens),
            str(r.completion_tokens),
            str(r.answer_words),
            status,
        )

    console.print(table)
    console.print()


def _print_conclusions(param_results: list[RunResult], prompt_results: list[RunResult]) -> None:
    lines: list[str] = []

    # Находим самый быстрый по tokens/sec из param_results без ошибок
    ok_params = [r for r in param_results if not r.error]
    if ok_params:
        fastest = max(ok_params, key=lambda r: r.tokens_per_sec)
        slowest = min(ok_params, key=lambda r: r.tokens_per_sec)
        lines.append(
            f"[bold cyan]Скорость:[/bold cyan] «{fastest.label}» быстрее всего "
            f"({fastest.tokens_per_sec:.1f} tok/sec) vs «{slowest.label}» "
            f"({slowest.tokens_per_sec:.1f} tok/sec)"
        )

        baseline = ok_params[0]
        optimal_candidates = [r for r in ok_params if r.label != baseline.label]
        if optimal_candidates:
            opt = min(optimal_candidates, key=lambda r: r.completion_tokens)
            savings = baseline.completion_tokens - opt.completion_tokens
            if savings > 0:
                lines.append(
                    f"[bold cyan]Токены:[/bold cyan] «{opt.label}» экономит ~{savings} output-токенов "
                    f"vs базового (меньше = дешевле/быстрее)"
                )

    # Качество по prompt-шаблонам — самый структурированный (больше слов = полнее)
    ok_prompts = [r for r in prompt_results if not r.error]
    if ok_prompts:
        richest = max(ok_prompts, key=lambda r: r.answer_words)
        lines.append(
            f"[bold cyan]Промпт:[/bold cyan] «{richest.label}» даёт наиболее развёрнутый ответ "
            f"({richest.answer_words} слов)"
        )
        lines.append(
            "[bold cyan]Рекомендация:[/bold cyan] специализированный промпт + "
            "temperature=0.2 — оптимальный баланс скорости и качества"
        )

    console.print(Panel(
        "\n".join(lines) if lines else "Нет данных для выводов.",
        title="[bold green]Итог оптимизации[/bold green]",
        border_style="green",
        padding=(0, 1),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Публичная точка входа
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = "Ты полезный ассистент. Отвечай на русском языке."


def run_local_optimize_demo(
    base_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model_id: str = "local",
) -> None:
    """Запустить демо оптимизации локальной LLM."""
    _print_header(base_url, model_id)

    with OpenRouterClient(api_key="local", base_url=base_url) as client:

        # ── Этап 1: параметры ───────────────────────────────────────────────
        console.print(Rule("[bold yellow]Этап 1 — Параметры (temperature + max_tokens)[/bold yellow]"))
        console.print()
        console.print(f"  [dim]Вопрос:[/dim] {_TEST_QUESTION}\n")

        param_results: list[RunResult] = []
        for cfg in PARAM_CONFIGS:
            console.print(
                f"  [cyan]→[/cyan] [bold]{cfg.label}[/bold]  "
                f"[dim](temp={cfg.temperature}, max_tokens={cfg.max_tokens})[/dim]"
            )
            with console.status("    Запрос к модели...", spinner="dots"):
                result = _run_single(
                    client=client,
                    model_id=model_id,
                    system_prompt=_DEFAULT_SYSTEM,
                    question=_TEST_QUESTION,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    label=cfg.label,
                    description=cfg.description,
                )
            param_results.append(result)
            console.print(
                f"    [green]✓[/green] {result.duration_ms} мс · "
                f"{result.tokens_per_sec:.1f} tok/sec · "
                f"{result.completion_tokens} output-токенов"
            )
            console.print()

        console.print()
        for r in param_results:
            _print_answer_preview(r)
            console.print()

        _print_metrics_table(param_results, "Этап 1: сравнение параметров")

        # ── Этап 2: prompt-шаблоны ──────────────────────────────────────────
        console.print(Rule("[bold yellow]Этап 2 — Prompt-шаблоны[/bold yellow]"))
        console.print()
        console.print(
            "  [dim]Параметры фиксированы:[/dim] "
            "[cyan]temperature=0.2, max_tokens=256[/cyan]\n"
        )

        prompt_results: list[RunResult] = []
        for cfg_p in PROMPT_CONFIGS:
            console.print(
                f"  [cyan]→[/cyan] [bold]{cfg_p.label}[/bold]  "
                f"[dim]{cfg_p.description}[/dim]"
            )
            with console.status("    Запрос к модели...", spinner="dots"):
                result = _run_single(
                    client=client,
                    model_id=model_id,
                    system_prompt=cfg_p.system_prompt,
                    question=_TEST_QUESTION,
                    temperature=0.2,
                    max_tokens=256,
                    label=cfg_p.label,
                    description=cfg_p.description,
                )
            prompt_results.append(result)
            console.print(
                f"    [green]✓[/green] {result.duration_ms} мс · "
                f"{result.tokens_per_sec:.1f} tok/sec · "
                f"{result.answer_words} слов"
            )
            console.print()

        console.print()
        for r in prompt_results:
            _print_answer_preview(r)
            console.print()

        _print_metrics_table(prompt_results, "Этап 2: сравнение prompt-шаблонов")

        # ── Итог ────────────────────────────────────────────────────────────
        _print_conclusions(param_results, prompt_results)
