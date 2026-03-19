"""Точка входа: python -m llm_cli."""

from __future__ import annotations

import argparse
import sys

from .api import OpenRouterClient
from .benchmark import run_benchmark
from .chat import run_chat, run_chat_with_tools
from .config import ensure_config
from .models import BENCHMARK_PROMPT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cli",
        description="LLM CLI Chat — терминальный клиент для OpenRouter API",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Запустить бенчмарк-сравнение моделей",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Промпт для бенчмарка (используется с --compare)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Температура генерации",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Подключиться к MCP-серверу и вывести список доступных инструментов",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Запустить чат с MCP tool calling (LLM сама вызывает инструменты)",
    )
    parser.add_argument(
        "--agent-demo",
        action="store_true",
        help="Демо: агент автоматически вызывает MCP-инструменты и получает результат",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Запустить фоновый планировщик 24/7 (сбор цен + напоминания)",
    )
    parser.add_argument(
        "--scheduler-demo",
        action="store_true",
        help="Автодемо планировщика: агент устанавливает напоминание, запускает мониторинг, получает сводку",
    )
    parser.add_argument(
        "--pipeline-demo",
        action="store_true",
        help="Автоматический пайплайн: search → summarize → save_to_file",
    )
    parser.add_argument(
        "--orchestration-demo",
        action="store_true",
        help="Оркестрация MCP: длинный флоу через два сервера (Data & Analytics + Tools & Storage)",
    )
    parser.add_argument(
        "--rag-index",
        action="store_true",
        help="Запустить RAG-пайплайн индексации документов (chunking + embeddings + FAISS)",
    )
    parser.add_argument(
        "--rag-strategy",
        choices=["fixed", "structural", "both"],
        default="both",
        help="Стратегия чанкинга для --rag-index (default: both)",
    )
    parser.add_argument(
        "--rag-no-compare",
        action="store_true",
        help="Пропустить сравнение стратегий при --rag-index",
    )
    parser.add_argument(
        "--rag-demo",
        action="store_true",
        help="Наглядная демонстрация результатов RAG-задания (использует готовый индекс)",
    )
    parser.add_argument(
        "--rag-chat",
        action="store_true",
        help="Интерактивный чат с двумя режимами: /rag on|off переключает RAG в реальном времени",
    )
    parser.add_argument(
        "--rag-demo-suite",
        action="store_true",
        help="Полный demo-suite: baseline/rewrite-only/improved + итоговая таблица",
    )
    parser.add_argument(
        "--rag-grounded-demo",
        action="store_true",
        help="Видео-демо: 10 вопросов с проверкой sources/citations + проверка режима 'не знаю'",
    )
    parser.add_argument(
        "--rag-eval",
        action="store_true",
        help="10 контрольных вопросов: side-by-side сравнение ответов без RAG и с RAG",
    )
    parser.add_argument(
        "--rag-compare",
        type=str,
        default=None,
        metavar="QUESTION_ID",
        help="Детальное сравнение без/с RAG для одного вопроса (ID 1–10 из eval или произвольный текст)",
    )
    parser.add_argument(
        "--rag-eval-strategy",
        choices=["fixed", "structural"],
        default="structural",
        help="Стратегия чанкинга для --rag-eval и --rag-chat (default: structural)",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=5,
        help="Число чанков для RAG-поиска (default: 5)",
    )
    parser.add_argument(
        "--rag-top-k-before",
        type=int,
        default=None,
        help="Сколько кандидатов извлекать из FAISS до post-retrieval (default: --rag-top-k)",
    )
    parser.add_argument(
        "--rag-top-k-after",
        type=int,
        default=None,
        help="Сколько чанков оставлять после фильтра/реранка (default: --rag-top-k-before)",
    )
    parser.add_argument(
        "--rag-min-similarity",
        type=float,
        default=0.0,
        help="Порог similarity для отсечения нерелевантных кандидатов (default: 0.0)",
    )
    parser.add_argument(
        "--rag-post-mode",
        choices=["off", "threshold", "rerank"],
        default="off",
        help="Post-retrieval режим: off, threshold или rerank",
    )
    parser.add_argument(
        "--rag-rewrite",
        choices=["on", "off"],
        default="on",
        help="Включить/выключить query rewrite перед retrieval (default: on)",
    )
    parser.add_argument(
        "--rag-question-limit",
        type=int,
        default=10,
        help="Сколько контрольных вопросов прогонять в --rag-demo-suite (default: 10)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mcp:
        from .mcp_client import run_mcp_demo
        run_mcp_demo()
        return

    if args.agent_demo:
        from .mcp_client import run_agent_demo
        run_agent_demo()
        return

    if args.scheduler:
        from .scheduler_daemon import run_scheduler_daemon
        run_scheduler_daemon()
        return

    if args.scheduler_demo:
        from .scheduler_daemon import run_scheduler_demo
        cfg = ensure_config()
        run_scheduler_demo(cfg.api_key)
        return

    if args.pipeline_demo:
        from .mcp_client import run_pipeline_demo
        run_pipeline_demo()
        return

    if args.orchestration_demo:
        from .mcp_client import run_orchestration_demo
        run_orchestration_demo()
        return

    if args.rag_index:
        from .rag.pipeline import run_pipeline
        run_pipeline(strategy=args.rag_strategy, compare=not args.rag_no_compare)
        return

    if args.rag_demo:
        from .rag.demo import run_demo
        run_demo()
        return

    if args.rag_chat or args.rag_eval or args.rag_demo_suite or args.rag_grounded_demo or args.rag_compare is not None:
        from .rag.rag_demo import (
            run_demo_suite,
            run_full_eval,
            run_grounded_demo,
            run_interactive_chat,
            run_single_comparison,
        )
        cfg = ensure_config()
        model = cfg.default_model
        strategy = args.rag_eval_strategy
        top_k = args.rag_top_k
        top_k_before = args.rag_top_k_before
        top_k_after = args.rag_top_k_after
        min_similarity = args.rag_min_similarity
        post_mode = args.rag_post_mode
        rewrite_enabled = args.rag_rewrite == "on"

        if args.rag_eval:
            run_full_eval(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                min_similarity=min_similarity,
                post_mode=post_mode,
                rewrite_enabled=rewrite_enabled,
            )
            return

        if args.rag_demo_suite:
            run_demo_suite(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                min_similarity=min_similarity,
                improved_post_mode=post_mode if post_mode != "off" else "threshold",
                question_limit=args.rag_question_limit,
            )
            return

        if args.rag_grounded_demo:
            run_grounded_demo(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                min_similarity=min_similarity if min_similarity > 0 else 0.45,
                post_mode=post_mode if post_mode != "off" else "threshold",
                rewrite_enabled=rewrite_enabled,
            )
            return

        if args.rag_compare is not None:
            # Если передан числовой ID — берём вопрос из EVAL_QUESTIONS, иначе — как текст
            try:
                q_id = int(args.rag_compare)
                run_single_comparison(
                    cfg.api_key,
                    model,
                    question_id=q_id,
                    strategy=strategy,
                    top_k=top_k,
                    top_k_before=top_k_before,
                    top_k_after=top_k_after,
                    min_similarity=min_similarity,
                    post_mode=post_mode,
                    rewrite_enabled=rewrite_enabled,
                )
            except ValueError:
                run_single_comparison(
                    cfg.api_key,
                    model,
                    question=args.rag_compare,
                    strategy=strategy,
                    top_k=top_k,
                    top_k_before=top_k_before,
                    top_k_after=top_k_after,
                    min_similarity=min_similarity,
                    post_mode=post_mode,
                    rewrite_enabled=rewrite_enabled,
                )
            return

        if args.rag_chat:
            run_interactive_chat(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                min_similarity=min_similarity,
                post_mode=post_mode,
                rewrite_enabled=rewrite_enabled,
            )
            return

    cfg = ensure_config()

    if args.tools:
        with OpenRouterClient(cfg.api_key) as client:
            run_chat_with_tools(client, cfg)
        return

    temperature = args.temp if args.temp is not None else cfg.temperature

    with OpenRouterClient(cfg.api_key) as client:
        if args.compare:
            prompt = args.prompt or cfg.benchmark_prompt or BENCHMARK_PROMPT
            run_benchmark(client, prompt, cfg.models, temperature)
        else:
            cfg.temperature = temperature
            run_chat(client, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
