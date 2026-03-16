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
