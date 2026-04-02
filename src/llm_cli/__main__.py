"""Точка входа: python -m llm_cli."""

from __future__ import annotations

import argparse
import sys

from .api import OpenRouterClient
from .benchmark import run_benchmark
from .chat import run_chat, run_chat_with_tools
from .config import ensure_config
from .models import BENCHMARK_PROMPT, LOCAL_BASE_URL, LOCAL_MODEL_ID, OPENROUTER_URL


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
    parser.add_argument(
        "--rag-memory-chat",
        action="store_true",
        help="Интерактивный RAG-чат с историей диалога и памятью задачи (task state)",
    )
    parser.add_argument(
        "--rag-memory-demo",
        action="store_true",
        help="Автодемо: два сценария по 11–12 сообщений с RAG + памятью задачи (для видео)",
    )
    parser.add_argument(
        "--rag-memory-pause",
        type=float,
        default=0.3,
        help="Пауза между сообщениями в автодемо (секунды, default: 0.3)",
    )
    parser.add_argument(
        "--rag-local-demo",
        action="store_true",
        help="Полностью локальный RAG: FAISS + MiniLM (retrieval) + llama.cpp :8081 (генерация). OpenRouter не нужен.",
    )
    parser.add_argument(
        "--rag-local-url",
        type=str,
        default="http://127.0.0.1:8081/v1/chat/completions",
        help="URL локального llama-server (default: http://127.0.0.1:8081/v1/chat/completions)",
    )
    parser.add_argument(
        "--rag-local-model",
        type=str,
        default="local",
        help="Идентификатор модели для локального сервера (default: local)",
    )
    parser.add_argument(
        "--support-demo",
        action="store_true",
        help=(
            "Демо ассистента поддержки пользователей: "
            "RAG (FAQ) + CRM (тикеты из JSON) + LLM-ответы по реальным тикетам"
        ),
    )
    parser.add_argument(
        "--local-optimize",
        action="store_true",
        help=(
            "Демо оптимизации локальной LLM: сравнение temperature/max_tokens и prompt-шаблонов "
            "(требует запущенного llama-server на --local-url)"
        ),
    )
    parser.add_argument(
        "--local-url",
        type=str,
        default="http://127.0.0.1:8081/v1/chat/completions",
        help="URL локального llama-server для --local-optimize (default: http://127.0.0.1:8081/v1/chat/completions)",
    )
    parser.add_argument(
        "--local-model-id",
        type=str,
        default="local",
        help="Идентификатор модели для --local-optimize (default: local)",
    )
    parser.add_argument(
        "--file-assistant",
        action="store_true",
        help=(
            "Интерактивный ассистент для работы с файлами проекта: "
            "чтение, поиск по коду, генерация документации, diff и запись файлов"
        ),
    )
    parser.add_argument(
        "--file-goal",
        type=str,
        default=None,
        metavar="GOAL",
        help=(
            "Выполнить одну задачу неинтерактивно (используется с --file-assistant): "
            "например: --file-goal 'найди все импорты httpx'"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.local_optimize:
        from .local_optimize import run_local_optimize_demo
        run_local_optimize_demo(base_url=args.local_url, model_id=args.local_model_id)
        return

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

    if args.rag_local_demo:
        from .rag.rag_demo import run_local_rag_demo
        run_local_rag_demo(
            base_url=args.rag_local_url,
            model_id=args.rag_local_model,
            strategy=args.rag_eval_strategy,
            top_k=args.rag_top_k,
            question_limit=args.rag_question_limit,
        )
        return

    if args.rag_memory_chat or args.rag_memory_demo:
        from .rag.memory_demo import run_memory_chat, run_memory_demo
        cfg = ensure_config()
        model = cfg.default_model
        strategy = args.rag_eval_strategy
        top_k = args.rag_top_k

        if args.rag_memory_chat:
            run_memory_chat(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
            )
        else:
            run_memory_demo(
                cfg.api_key,
                model,
                strategy=strategy,
                top_k=top_k,
                pause=args.rag_memory_pause,
            )
        return

    if args.support_demo:
        from .support_assistant import run_support_demo
        cfg = ensure_config()
        run_support_demo(api_key=cfg.api_key, model=cfg.default_model)
        return

    if args.file_assistant:
        from .file_assistant import run_file_assistant, run_file_assistant_goal
        from .models import LOCAL_MODEL_ID
        cfg = ensure_config()
        if cfg.use_local:
            _fa_api_key = "local"
            _fa_base_url = cfg.local_url
            _fa_model = LOCAL_MODEL_ID
        else:
            _fa_api_key = cfg.api_key
            _fa_base_url = OPENROUTER_URL
            _fa_model = cfg.default_model
        if args.file_goal:
            run_file_assistant_goal(
                api_key=_fa_api_key,
                goal=args.file_goal,
                model=_fa_model,
                base_url=_fa_base_url,
            )
        else:
            run_file_assistant(
                api_key=_fa_api_key,
                model=_fa_model,
                base_url=_fa_base_url,
            )
        return

    cfg = ensure_config()

    if cfg.use_local:
        _base_url = cfg.local_url
        _api_key = "local"
        cfg.default_model = LOCAL_MODEL_ID
    else:
        _base_url = OPENROUTER_URL
        _api_key = cfg.api_key

    if args.tools:
        with OpenRouterClient(_api_key, base_url=_base_url) as client:
            run_chat_with_tools(client, cfg)
        return

    temperature = args.temp if args.temp is not None else cfg.temperature

    with OpenRouterClient(_api_key, base_url=_base_url) as client:
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
