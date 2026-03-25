"""Демо-скрипт для видео: агент с двумя режимами (с RAG / без RAG).

Режимы:
  - run_single_comparison() — детальный side-by-side для одного вопроса
  - run_full_eval()         — 10 контрольных вопросов с итоговой таблицей
  - run_local_rag_demo()    — полностью локальный RAG (llama.cpp + FAISS + MiniLM)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import cast

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box
from rich.table import Table

from .eval import EVAL_QUESTIONS, EvalMode, run_eval, run_mode_comparison_eval
from .rag_agent import RagAgent
from .relevance import PostRetrievalMode

console = Console()

_PROJECT_ROOT = Path(__file__).parents[3]
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"


def _header() -> None:
    console.print()
    console.print(Panel(
        Text.assemble(
            ("RAG Agent Demo", "bold cyan"),
            "\n",
            ("HuggingFace NLP Course  ·  Structural Chunking  ·  FAISS IndexFlatIP", "dim"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def run_single_comparison(
    api_key: str,
    model: str,
    question: str | None = None,
    question_id: int = 1,
    strategy: str = "structural",
    top_k: int = 5,
    top_k_before: int | None = None,
    top_k_after: int | None = None,
    min_similarity: float = 0.0,
    post_mode: PostRetrievalMode = "off",
    rewrite_enabled: bool = True,
) -> None:
    """Сравнить ответы без RAG и с RAG для одного вопроса.

    Args:
        api_key:     Ключ OpenRouter.
        model:       Идентификатор модели.
        question:    Текст вопроса (если None — берётся вопрос question_id из EVAL_QUESTIONS).
        question_id: Номер вопроса из EVAL_QUESTIONS (1–10), используется когда question=None.
        strategy:    Стратегия чанкинга: "fixed" или "structural".
        top_k:       Количество извлекаемых чанков.
    """
    _header()

    q_data = next((q for q in EVAL_QUESTIONS if q["id"] == question_id), EVAL_QUESTIONS[0])
    text = question or q_data["question"]

    console.print(Rule("[bold]Single Question Comparison[/bold]"))
    console.print(f"\n[bold yellow]Вопрос:[/bold yellow] {text}\n")

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        min_similarity=min_similarity,
        post_retrieval_mode=post_mode,
        rewrite_enabled=rewrite_enabled,
    )

    # --- Без RAG ---
    console.print("[bold red][ БЕЗ RAG ][/bold red] Отправляем запрос к LLM...")
    t0 = time.perf_counter()
    ans_no_rag = agent.ask(text, use_rag=False)
    elapsed_no_rag = time.perf_counter() - t0
    console.print(f"[dim]Готово за {elapsed_no_rag:.1f}с[/dim]")

    # --- С RAG ---
    console.print("\n[bold green][ С RAG ][/bold green] Ищем релевантные чанки...")
    t0 = time.perf_counter()
    ans_rag = agent.ask(text, use_rag=True)
    elapsed_rag = time.perf_counter() - t0
    console.print(f"[dim]Готово за {elapsed_rag:.1f}с[/dim]\n")

    # --- Вывод источников ---
    if ans_rag.chunks:
        console.print(Rule("[dim]Найденные источники[/dim]"))
        src_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        src_tbl.add_column("#", width=3, justify="right")
        src_tbl.add_column("Файл / Раздел")
        src_tbl.add_column("Score", justify="right", width=8)
        src_tbl.add_column("Текст (начало)", ratio=1)

        for i, (chunk, score) in enumerate(ans_rag.chunks, 1):
            src = chunk.metadata.get("file", chunk.metadata.get("source", "?"))
            section = chunk.metadata.get("section", "")
            snippet = chunk.text.strip().replace("\n", " ")[:80] + "…"
            src_tbl.add_row(str(i), f"{src} · {section}" if section else src, f"{score:.3f}", snippet)

        console.print(src_tbl)

    # --- Side-by-side панели ---
    console.print()
    console.print(Rule("[bold]Сравнение ответов[/bold]"))
    console.print()

    panel_no_rag = Panel(
        ans_no_rag.text.strip(),
        title="[bold red]БЕЗ RAG[/bold red]",
        border_style="red",
        padding=(1, 2),
    )
    panel_rag = Panel(
        ans_rag.text.strip(),
        title=f"[bold green]С RAG ({len(ans_rag.chunks)} источников)[/bold green]",
        subtitle=f"[dim]{', '.join(ans_rag.sources[:3])}[/dim]" if ans_rag.sources else None,
        border_style="green",
        padding=(1, 2),
    )

    # Rich Columns для side-by-side при достаточной ширине терминала
    width = console.width
    if width >= 120:
        console.print(Columns([panel_no_rag, panel_rag], equal=True, expand=True))
    else:
        console.print(panel_no_rag)
        console.print()
        console.print(panel_rag)

    # --- Мини-оценка по ключевым словам ---
    console.print()
    expected_kw = q_data.get("expected_keywords", [])
    if expected_kw:
        no_rag_hits = sum(1 for kw in expected_kw if kw.lower() in ans_no_rag.text.lower())
        rag_hits = sum(1 for kw in expected_kw if kw.lower() in ans_rag.text.lower())
        console.print(
            f"[dim]Ключевые слова ({', '.join(expected_kw)}): "
            f"без RAG — {no_rag_hits}/{len(expected_kw)}, "
            f"с RAG — {rag_hits}/{len(expected_kw)}[/dim]"
        )
        if rag_hits > no_rag_hits:
            console.print("[bold green]✓ RAG добавил полезный контекст[/bold green]")
        elif rag_hits == no_rag_hits:
            console.print("[yellow]≈ Результаты сопоставимы[/yellow]")
        else:
            console.print("[red]✗ Без RAG ответ содержит больше ключевых слов[/red]")

    if ans_rag.retrieval_stats is not None:
        st = ans_rag.retrieval_stats
        console.print(
            "[dim]"
            f"RAG mode={st.mode}, rewrite={'yes' if st.query_rewritten else 'no'}, "
            f"top_k(before/after)={st.top_k_before}/{st.top_k_after}, "
            f"min_similarity={st.min_similarity:.3f}, "
            f"selected={st.selected_count}, fallback={'yes' if st.fallback_used else 'no'}"
            "[/dim]"
        )

    console.print()


def run_full_eval(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
    top_k_before: int | None = None,
    top_k_after: int | None = None,
    min_similarity: float = 0.0,
    post_mode: PostRetrievalMode = "off",
    rewrite_enabled: bool = True,
) -> None:
    """Запустить полный eval: 10 контрольных вопросов с итоговой таблицей.

    Args:
        api_key:  Ключ OpenRouter.
        model:    Идентификатор модели.
        strategy: Стратегия чанкинга: "fixed" или "structural".
        top_k:    Количество извлекаемых чанков на вопрос.
    """
    _header()

    console.print(f"[dim]Модель: {model}  ·  Стратегия: {strategy}  ·  top_k={top_k}[/dim]")
    console.print()

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        min_similarity=min_similarity,
        post_retrieval_mode=post_mode,
        rewrite_enabled=rewrite_enabled,
    )

    # Прогреть индекс до начала eval
    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс загружен: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    run_eval(agent)


def run_interactive_chat(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
    top_k_before: int | None = None,
    top_k_after: int | None = None,
    min_similarity: float = 0.0,
    post_mode: PostRetrievalMode = "off",
    rewrite_enabled: bool = True,
) -> None:
    """Интерактивный режим чата с переключением RAG on/off командой /rag.

    Команды:
      /rag on   — включить режим RAG
      /rag off  — выключить режим RAG
      /quit     — выйти
    """
    _header()

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        min_similarity=min_similarity,
        post_retrieval_mode=post_mode,
        rewrite_enabled=rewrite_enabled,
    )

    use_rag = True

    console.print(Panel(
        "[bold]Интерактивный RAG-чат[/bold]\n\n"
        "Команды:\n"
        "  [cyan]/rag on[/cyan]   — включить режим RAG (по умолчанию)\n"
        "  [cyan]/rag off[/cyan]  — выключить RAG\n"
        "  [cyan]/quit[/cyan]     — выйти\n\n"
        "База знаний: HuggingFace NLP Course (главы 1–5)",
        border_style="blue",
    ))
    console.print()

    while True:
        mode_badge = "[green][RAG][/green]" if use_rag else "[red][no-RAG][/red]"
        try:
            user_input = console.input(f"{mode_badge} [bold]Вопрос:[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Выход.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Выход.[/dim]")
            break

        if user_input.lower() == "/rag on":
            use_rag = True
            console.print("[green]Режим RAG включён[/green]\n")
            continue

        if user_input.lower() == "/rag off":
            use_rag = False
            console.print("[red]Режим RAG выключен[/red]\n")
            continue

        console.print(f"[dim]Запрос {'с RAG' if use_rag else 'без RAG'}...[/dim]")
        t0 = time.perf_counter()
        answer = agent.ask(user_input, use_rag=use_rag)
        elapsed = time.perf_counter() - t0

        if use_rag and answer.chunks:
            sources_str = ", ".join(answer.sources[:3])
            console.print(f"[dim]Источники: {sources_str}[/dim]")

        console.print(
            Panel(
                answer.text.strip(),
                title=f"[{'green' if use_rag else 'red'}]{'С RAG' if use_rag else 'БЕЗ RAG'}[/{'green' if use_rag else 'red'}]"
                      f"  [dim]{elapsed:.1f}с[/dim]",
                border_style="green" if use_rag else "red",
            )
        )
        console.print()


def run_demo_suite(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
    top_k_before: int | None = None,
    top_k_after: int | None = None,
    min_similarity: float = 0.45,
    improved_post_mode: PostRetrievalMode = "threshold",
    question_limit: int = 10,
) -> None:
    """Единая демо-команда: baseline/rewrite-only/improved + итоговая сводка."""
    _header()

    before_k = top_k_before if top_k_before is not None else top_k
    after_k = top_k_after if top_k_after is not None else before_k

    console.print(
        f"[dim]Модель: {model} · Стратегия: {strategy} · "
        f"top_k(before/after)={before_k}/{after_k} · "
        f"improved_mode={improved_post_mode} · min_similarity={min_similarity:.3f}[/dim]\n"
    )

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        top_k_before=before_k,
        top_k_after=after_k,
        min_similarity=min_similarity,
        post_retrieval_mode="off",
        rewrite_enabled=True,
    )

    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс загружен: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    modes = [
        EvalMode(
            name="baseline",
            label="[yellow]baseline (no-rewrite, no-filter)[/yellow]",
            rewrite_enabled=False,
            post_mode="off",
            top_k_before=before_k,
            top_k_after=after_k,
            min_similarity=0.0,
        ),
        EvalMode(
            name="rewrite_only",
            label="[cyan]rewrite-only[/cyan]",
            rewrite_enabled=True,
            post_mode="off",
            top_k_before=before_k,
            top_k_after=after_k,
            min_similarity=0.0,
        ),
        EvalMode(
            name="improved",
            label="[green]improved (rewrite + filter/rerank)[/green]",
            rewrite_enabled=True,
            post_mode=cast(PostRetrievalMode, improved_post_mode),
            top_k_before=before_k,
            top_k_after=after_k,
            min_similarity=min_similarity,
        ),
    ]

    run_mode_comparison_eval(agent, modes=modes, question_limit=question_limit)


def run_grounded_demo(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
    top_k_before: int | None = None,
    top_k_after: int | None = None,
    min_similarity: float = 0.45,
    post_mode: PostRetrievalMode = "threshold",
    rewrite_enabled: bool = True,
) -> None:
    """Демо для видео: обязательные источники/цитаты + режим "не знаю"."""
    _header()
    before_k = top_k_before if top_k_before is not None else top_k
    after_k = top_k_after if top_k_after is not None else before_k

    console.print(
        f"[dim]Grounded RAG Demo · model={model} · strategy={strategy} · "
        f"mode={post_mode} · min_similarity={min_similarity:.3f} · top_k={before_k}/{after_k}[/dim]\n"
    )

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        top_k_before=before_k,
        top_k_after=after_k,
        min_similarity=min_similarity,
        post_retrieval_mode=post_mode,
        rewrite_enabled=rewrite_enabled,
    )

    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс загружен: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    # 1) Обязательные источники/цитаты и проверка поддержки смысла на 10 вопросах
    run_eval(agent)

    # 2) Демонстрация режима "не знаю" на заведомо слабом запросе
    console.rule("[bold]Проверка режима 'не знаю'[/bold]")
    weak_question = "Какой официальный размер рынка квантовых аккумуляторов в 2035 году?"
    console.print(f"[bold yellow]Вопрос:[/bold yellow] {weak_question}\n")

    answer = agent.ask(
        weak_question,
        use_rag=True,
        rewrite_enabled=rewrite_enabled,
        post_retrieval_mode=post_mode,
        top_k_before=before_k,
        top_k_after=after_k,
        min_similarity=min_similarity,
    )
    console.print(
        Panel(
            answer.text.strip(),
            title="[bold green]RAG ответ[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    if answer.unknown_due_to_low_relevance:
        console.print("[bold green]✓ Режим 'не знаю' сработал корректно[/bold green]")
    else:
        console.print("[bold red]✗ Режим 'не знаю' не сработал — увеличьте --rag-min-similarity[/bold red]")


# ---------------------------------------------------------------------------
# Локальная LLM + RAG (llama.cpp на порту 8081)
# ---------------------------------------------------------------------------

_LOCAL_DEMO_QUESTIONS = [
    {
        "id": 1,
        "question": "Что такое токенизатор и зачем он нужен в NLP?",
        "expected_keywords": ["токен", "token", "subword", "tokenizer", "текст", "text"],
        "hint": "Базовое понятие — как текст превращается в числа для модели",
    },
    {
        "id": 5,
        "question": "Что такое fine-tuning и чем он отличается от pre-training предобученной модели?",
        "expected_keywords": ["fine-tuning", "fine-tune", "pre-train", "pretrain", "задача", "task"],
        "hint": "Как переиспользуют предобученные трансформеры для конкретных задач",
    },
    {
        "id": 8,
        "question": "Что такое BERT и в чём его архитектурные особенности по сравнению с GPT?",
        "expected_keywords": ["bert", "encoder", "bidirectional", "masked", "mlm"],
        "hint": "Энкодерная модель с двунаправленным вниманием и предобучением на MLM",
    },
]


def run_local_rag_demo(
    base_url: str = "http://127.0.0.1:8081/v1/chat/completions",
    model_id: str = "local",
    strategy: str = "structural",
    top_k: int = 5,
    question_limit: int = 3,
) -> None:
    """Полностью локальный RAG-демо: FAISS + MiniLM (retrieval) + llama.cpp (генерация).

    API-ключ OpenRouter не нужен. Локальная модель должна быть запущена на base_url.
    """
    console.print()
    console.print(Panel(
        Text.assemble(
            ("Local LLM + RAG Demo", "bold magenta"),
            "\n",
            ("FAISS · all-MiniLM-L6-v2 · llama.cpp (Qwen3.5-27B)", "dim"),
            "\n",
            (f"Эндпоинт: {base_url}", "dim cyan"),
        ),
        border_style="magenta",
        padding=(1, 4),
    ))
    console.print()

    agent = RagAgent(
        api_key="local",
        model=model_id,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        rewrite_enabled=False,
        temperature=0.3,
        base_url=base_url,
    )

    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс загружен: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    questions = _LOCAL_DEMO_QUESTIONS[:max(1, question_limit)]

    summary_rows: list[tuple[str, str, str, str, str]] = []

    for idx, q in enumerate(questions, 1):
        console.rule(f"[bold]Вопрос {idx}/{len(questions)}[/bold]")
        console.print(f"[bold yellow]{q['question']}[/bold yellow]")
        console.print(f"[dim]{q['hint']}[/dim]\n")

        # --- Без RAG ---
        console.print("[bold red][ БЕЗ RAG ][/bold red] Запрос к локальной модели...")
        t0 = time.perf_counter()
        ans_no_rag = agent.ask(q["question"], use_rag=False)
        elapsed_no_rag = time.perf_counter() - t0
        console.print(f"[dim]Готово за {elapsed_no_rag:.1f}с[/dim]\n")

        # --- С RAG ---
        console.print("[bold green][ С RAG ][/bold green] Ищем релевантные чанки в FAISS...")
        t0 = time.perf_counter()
        ans_rag = agent.ask(q["question"], use_rag=True)
        elapsed_rag = time.perf_counter() - t0
        console.print(f"[dim]Готово за {elapsed_rag:.1f}с[/dim]\n")

        # Найденные источники
        if ans_rag.chunks:
            src_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
            src_tbl.add_column("#", width=3, justify="right")
            src_tbl.add_column("Файл / Раздел")
            src_tbl.add_column("Score", justify="right", width=8)
            src_tbl.add_column("Текст (начало)", ratio=1)
            for i, (chunk, score) in enumerate(ans_rag.chunks, 1):
                src = chunk.metadata.get("file", chunk.metadata.get("source", "?"))
                section = chunk.metadata.get("section", "")
                snippet = chunk.text.strip().replace("\n", " ")[:80] + "…"
                src_tbl.add_row(str(i), f"{src} · {section}" if section else src, f"{score:.3f}", snippet)
            console.print(Rule("[dim]Найденные источники[/dim]"))
            console.print(src_tbl)
            console.print()

        # Side-by-side панели
        console.print(Rule("[bold]Сравнение ответов[/bold]"))
        console.print()
        panel_no_rag = Panel(
            ans_no_rag.text.strip(),
            title=f"[bold red]БЕЗ RAG[/bold red]  [dim]{elapsed_no_rag:.1f}с[/dim]",
            border_style="red",
            padding=(1, 2),
        )
        panel_rag = Panel(
            ans_rag.answer.strip() if ans_rag.answer else ans_rag.text.strip(),
            title=f"[bold green]С RAG ({len(ans_rag.chunks)} источников)[/bold green]  [dim]{elapsed_rag:.1f}с[/dim]",
            subtitle=f"[dim]{', '.join(ans_rag.sources[:3])}[/dim]" if ans_rag.sources else None,
            border_style="green",
            padding=(1, 2),
        )
        if console.width >= 120:
            console.print(Columns([panel_no_rag, panel_rag], equal=True, expand=True))
        else:
            console.print(panel_no_rag)
            console.print()
            console.print(panel_rag)

        # Мини-оценка по ключевым словам (только по тексту ответа модели)
        kw = q["expected_keywords"]
        hits_no_rag = sum(1 for k in kw if k.lower() in ans_no_rag.text.lower())
        hits_rag = sum(1 for k in kw if k.lower() in (ans_rag.answer or ans_rag.text).lower())
        console.print()
        if hits_rag > hits_no_rag:
            verdict = "[bold green]✓ RAG добавил полезный контекст[/bold green]"
        elif hits_rag == hits_no_rag:
            verdict = "[yellow]≈ Результаты сопоставимы[/yellow]"
        else:
            verdict = "[red]✗ Без RAG больше ключевых слов[/red]"
        console.print(
            f"[dim]Ключевые слова: без RAG — {hits_no_rag}/{len(kw)}, с RAG — {hits_rag}/{len(kw)}[/dim]  {verdict}"
        )
        console.print()

        avg_score = (
            sum(s for _, s in ans_rag.chunks) / len(ans_rag.chunks)
            if ans_rag.chunks else 0.0
        )
        summary_rows.append((
            q["question"][:60] + ("…" if len(q["question"]) > 60 else ""),
            f"{hits_no_rag}/{len(kw)}",
            f"{hits_rag}/{len(kw)}",
            f"{elapsed_no_rag:.1f}с",
            f"{elapsed_rag:.1f}с",
            f"{len(ans_rag.chunks)} · {avg_score:.2f}",
        ))

    # Итоговая таблица
    console.rule("[bold magenta]Итоги Local LLM + RAG[/bold magenta]")
    console.print()
    tbl = Table(
        title=f"Локальная модель: {model_id}  ·  Retrieval: FAISS structural",
        box=box.HEAVY_HEAD,
        expand=True,
    )
    tbl.add_column("Вопрос", ratio=3)
    tbl.add_column("Без RAG\n(hit)", justify="center", width=10)
    tbl.add_column("С RAG\n(hit)", justify="center", width=10)
    tbl.add_column("Время\nбез RAG", justify="center", width=10)
    tbl.add_column("Время\nс RAG", justify="center", width=10)
    tbl.add_column("Источники\n(n · avg_sim)", justify="center", width=16)

    rag_wins = 0
    rag_ties = 0
    for row in summary_rows:
        q_text, h_no, h_rag, t_no, t_rag, src_info = row
        h_no_int = int(h_no.split("/")[0])
        h_rag_int = int(h_rag.split("/")[0])
        if h_rag_int > h_no_int:
            rag_wins += 1
            rag_markup = f"[bold green]{h_rag}[/bold green]"
        elif h_rag_int == h_no_int:
            rag_ties += 1
            rag_markup = f"[yellow]{h_rag}[/yellow]"
        else:
            rag_markup = f"[red]{h_rag}[/red]"
        tbl.add_row(q_text, h_no, rag_markup, t_no, t_rag, f"[cyan]{src_info}[/cyan]")

    console.print(tbl)
    console.print()

    # Интерпретация результата для видео
    n = len(summary_rows)
    if rag_wins > 0:
        verdict_text = (
            f"[bold green]RAG улучшил ответ в {rag_wins}/{n} случаях[/bold green]  "
            f"[dim](по ключевым словам)[/dim]"
        )
    elif rag_ties == n:
        verdict_text = (
            f"[bold yellow]Результаты сопоставимы ({n}/{n} ничья)[/bold yellow]  "
            f"[dim]Qwen3.5-27B хорошо знает материал сам по себе — RAG добавляет точные источники[/dim]"
        )
    else:
        verdict_text = (
            f"[bold yellow]Без RAG — больше терминов в ответе[/bold yellow]  "
            f"[dim]Qwen3.5-27B сильная модель: RAG ценен точностью источников, а не числом слов[/dim]"
        )
    console.print(verdict_text)
    console.print()
    console.print()
    console.print(Panel(
        "[bold]Архитектура полностью локального RAG:[/bold]\n\n"
        f"  Вопрос → [cyan]all-MiniLM-L6-v2[/cyan] (embedder, локально)\n"
        f"          → [cyan]FAISS IndexFlatIP[/cyan] (data/index/{strategy})\n"
        f"          → top-{top_k} чанков → промпт\n"
        f"          → [magenta]llama-server :8081[/magenta] (Qwen3.5-27B, локально)\n"
        f"          → Ответ\n\n"
        "[dim]OpenRouter не используется. Всё работает офлайн.[/dim]",
        border_style="magenta",
        padding=(1, 3),
    ))
    console.print()
