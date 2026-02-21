"""Логика бенчмарка моделей и оценки качества моделью-судьёй."""

from __future__ import annotations

import time

from rich.console import Console

from .api import OpenRouterClient
from .display import (
    print_benchmark_header,
    print_comparison_table,
    print_conclusions,
    print_judge_header,
    print_judge_verdict,
    print_model_header,
    print_model_result,
)
from .models import (
    JUDGE_MODEL,
    BenchmarkResult,
    ChatMessage,
    ModelConfig,
)

console = Console()


def run_benchmark(
    client: OpenRouterClient,
    prompt: str,
    models: list[ModelConfig],
    temperature: float | None = None,
) -> None:
    messages = [ChatMessage(role="user", content=prompt)]
    results: list[BenchmarkResult] = []

    print_benchmark_header(prompt, temperature)

    for model in models:
        print_model_header(model)

        try:
            with console.status(
                f"[bold cyan]Запрос к {model.display_name}...[/bold cyan]",
                spinner="dots",
            ):
                start = time.perf_counter_ns()
                chat_response = client.send_raw(messages, model.id, temperature)
                duration_ms = (time.perf_counter_ns() - start) // 1_000_000

            if chat_response.error is not None:
                result = BenchmarkResult(
                    model=model,
                    duration_ms=duration_ms,
                    error=chat_response.error.message,
                )
                results.append(result)
                print_model_result(result)
                continue

            content = (
                chat_response.choices[0].message.content
                if chat_response.choices
                else "(пустой ответ)"
            )
            usage = chat_response.usage
            cost = BenchmarkResult.calculate_cost(usage, model)

            result = BenchmarkResult(
                model=model,
                response=content,
                usage=usage,
                duration_ms=duration_ms,
                cost_usd=cost,
            )
            results.append(result)
            print_model_result(result)

        except Exception as e:
            result = BenchmarkResult(model=model, error=str(e))
            results.append(result)
            print_model_result(result)

    print_comparison_table(results)
    print_conclusions(results)
    _run_judge(client, prompt, results)


def _run_judge(
    client: OpenRouterClient,
    original_prompt: str,
    results: list[BenchmarkResult],
) -> None:
    successful = [r for r in results if r.error is None]
    if len(successful) < 2:
        return

    print_judge_header(JUDGE_MODEL.display_name, JUDGE_MODEL.id)

    answers_block = "\n\n".join(
        f"--- {r.model.display_name} ({r.model.tier}) ---\n{r.response}"
        for r in successful
    )

    judge_prompt = (
        f"Ты — эксперт-оценщик ответов языковых моделей. "
        f"Тебе дан один и тот же вопрос и ответы от {len(successful)} разных моделей.\n\n"
        f'Исходный вопрос: "{original_prompt}"\n\n'
        f"Ответы моделей:\n{answers_block}\n\n"
        "Оцени каждый ответ по критериям:\n"
        "1. Точность (фактическая правильность)\n"
        "2. Полнота (насколько полно раскрыта тема)\n"
        "3. Качество языка (грамотность, нет ли артефактов, мусорных символов, смешения языков)\n"
        "4. Следование инструкции (уложился ли в 3-5 предложений)\n\n"
        "Дай краткую оценку каждой модели (1-2 предложения) и назови лучший ответ. Отвечай на русском."
    )

    try:
        messages = [ChatMessage(role="user", content=judge_prompt)]

        with console.status(
            "[bold magenta]Судья оценивает ответы...[/bold magenta]",
            spinner="dots",
        ):
            start = time.perf_counter_ns()
            response = client.send_raw(messages, JUDGE_MODEL.id, 0.3)
            duration_ms = (time.perf_counter_ns() - start) // 1_000_000

        if response.error is not None:
            console.print(f"  [red]Ошибка судьи: {response.error.message}[/red]")
            return

        verdict = (
            response.choices[0].message.content
            if response.choices
            else "(пустой ответ)"
        )
        usage = response.usage
        cost = BenchmarkResult.calculate_cost(usage, JUDGE_MODEL)

        print_judge_verdict(
            verdict=verdict,
            duration_ms=duration_ms,
            usage_prompt=usage.prompt_tokens if usage else None,
            usage_completion=usage.completion_tokens if usage else None,
            usage_total=usage.total_tokens if usage else None,
            cost=cost,
        )

    except Exception as e:
        console.print(f"  [red]Ошибка при вызове судьи: {e}[/red]")
        console.print()
