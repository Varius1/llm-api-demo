"""Демо-скрипт для задания: мини-чат с RAG + историей + памятью задачи.

Два режима:
  run_memory_chat()  — интерактивный чат (ввод с клавиатуры)
  run_memory_demo()  — автоматические сценарии (без ввода, для записи видео)
"""
from __future__ import annotations

import time
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .memory_chat import RagMemoryChat, TaskMemory
from .rag_agent import RagAgent

console = Console()

_PROJECT_ROOT = Path(__file__).parents[3]
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"

# ─────────────────────────────────────────────────────────────────────────────
# Автосценарии
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_1: list[str] = [
    "Хочу разобраться в токенизации — с чего начать?",
    "Что такое BPE и чем он отличается от WordPiece?",
    "Как понять, какой токенизатор уже используется в модели?",
    "Можно ли обучить свой токенизатор на кастомном датасете?",
    "Что такое special tokens и зачем они нужны?",
    "Как добавить свои специальные токены в существующий токенизатор?",
    "Что такое Fast Tokenizer и в чём его преимущество?",
    "Объясни разницу между padding и truncation при батчевой обработке.",
    "Как работает attention_mask и зачем он нужен LLM?",
    "Что происходит, если входная последовательность длиннее max_length?",
    "Подведи итог: что нужно знать про токенизацию для fine-tuning модели?",
]

_SCENARIO_2: list[str] = [
    "Хочу сделать fine-tuning BERT для классификации текста — с чего начать?",
    "Какой датасет лучше взять для начального эксперимента?",
    "Как загрузить датасет и подготовить его к обучению?",
    "Как правильно токенизировать тексты для классификации?",
    "Что такое Trainer API и как его использовать?",
    "Какие гиперпараметры важно подобрать: learning rate, batch size?",
    "Как настроить TrainingArguments для начального запуска?",
    "Какие метрики использовать для оценки классификатора?",
    "Как передать метрики в Trainer через compute_metrics?",
    "Как сохранить обученную модель и загрузить её для инференса?",
    "Как применить pipeline для предсказания на новых текстах?",
    "Подведи итог: полный пайплайн fine-tuning BERT для классификации.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции отрисовки
# ─────────────────────────────────────────────────────────────────────────────


def _print_header() -> None:
    console.print()
    console.print(Panel(
        Text.assemble(
            ("RAG Chat + Memory", "bold cyan"),
            "\n",
            ("История диалога · TaskMemory · Источники при каждом ответе", "dim"),
            "\n",
            ("HuggingFace NLP Course  ·  FAISS IndexFlatIP  ·  Structural Chunking", "dim"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _print_task_memory(memory: TaskMemory, turn: int) -> None:
    if memory.is_empty():
        return
    lines: list[str] = []
    if memory.goal:
        lines.append(f"[bold]Цель:[/bold] {memory.goal}")
    if memory.constraints:
        c_str = ", ".join(f"{k}={v}" for k, v in memory.constraints.items())
        lines.append(f"[bold]Ограничения:[/bold] {c_str}")
    if memory.clarifications:
        cl_str = " | ".join(memory.clarifications[-3:])
        lines.append(f"[bold]Уточнения:[/bold] {cl_str}")
    lines.append(f"[dim]Ход {turn}[/dim]")
    console.print(Panel(
        "\n".join(lines),
        title="[bold yellow]Память задачи[/bold yellow]",
        border_style="yellow",
        padding=(0, 2),
        expand=False,
    ))


def _print_answer(answer_obj: object) -> None:
    from .memory_chat import RagMemoryAnswer
    assert isinstance(answer_obj, RagMemoryAnswer)

    console.print(Panel(
        answer_obj.answer,
        title=f"[bold green]Ответ  [dim](ход {answer_obj.turn}, {answer_obj.elapsed:.1f}с)[/dim][/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    if answer_obj.rewritten_query:
        console.print(
            f"  [dim]Запрос переписан для поиска: «{answer_obj.rewritten_query}»[/dim]"
        )

    if answer_obj.sources:
        src_tbl = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            padding=(0, 1),
        )
        src_tbl.add_column("#", width=3, justify="right")
        src_tbl.add_column("Файл / Раздел", ratio=2)
        src_tbl.add_column("Score", width=7, justify="right")
        src_tbl.add_column("Чанк", width=28)

        for i, ref in enumerate(answer_obj.sources, 1):
            src = ref.get("source", "?")
            section = ref.get("section", "")
            score = ref.get("score", "")
            chunk_id = ref.get("chunk_id", "")
            label = f"{src} · {section}" if section else src
            src_tbl.add_row(
                str(i),
                label,
                f"[cyan]{score}[/cyan]",
                f"[dim]{chunk_id}[/dim]",
            )
        console.print(
            Panel(
                src_tbl,
                title="[bold]Источники[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    else:
        console.print("[dim]  Источники: не найдено релевантных чанков[/dim]")

    console.print()


def _print_history_summary(chat: RagMemoryChat) -> None:
    hist = chat.history
    if not hist:
        return
    console.print(Rule("[bold dim]История диалога[/bold dim]"))
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    tbl.add_column("#", width=3, justify="right")
    tbl.add_column("Вопрос", ratio=1)
    tbl.add_column("Ответ (начало)", ratio=2)
    for i, turn in enumerate(hist, 1):
        q = turn["user"][:60] + ("…" if len(turn["user"]) > 60 else "")
        a = turn["assistant"][:100].replace("\n", " ") + "…"
        tbl.add_row(str(i), q, a)
    console.print(tbl)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Интерактивный чат
# ─────────────────────────────────────────────────────────────────────────────


def run_memory_chat(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
) -> None:
    """Интерактивный RAG-чат с историей диалога и памятью задачи.

    Команды:
      /memory   — показать текущую память задачи
      /history  — показать историю диалога
      /reset    — сбросить историю и память
      /quit     — выйти
    """
    _print_header()

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        rewrite_enabled=True,
    )

    chat = RagMemoryChat(agent=agent)

    console.print(Panel(
        "[bold]Интерактивный RAG-чат с памятью задачи[/bold]\n\n"
        "Команды:\n"
        "  [cyan]/memory[/cyan]   — показать текущую память задачи\n"
        "  [cyan]/history[/cyan]  — показать историю диалога\n"
        "  [cyan]/reset[/cyan]    — сбросить историю и память\n"
        "  [cyan]/quit[/cyan]     — выйти\n\n"
        "База знаний: [yellow]HuggingFace NLP Course[/yellow] (главы 1–5)",
        border_style="blue",
        expand=False,
    ))
    console.print()

    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс готов: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    while True:
        turn_label = f"[dim]ход {chat.task_memory.turn_count + 1}[/dim]"
        try:
            user_input = console.input(f"{turn_label} [bold cyan]Вопрос:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Выход.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Выход.[/dim]")
            break

        if user_input.lower() == "/memory":
            _print_task_memory(chat.task_memory, chat.task_memory.turn_count)
            continue

        if user_input.lower() == "/history":
            _print_history_summary(chat)
            continue

        if user_input.lower() == "/reset":
            chat.reset()
            console.print("[yellow]История и память задачи сброшены.[/yellow]\n")
            continue

        console.print("[dim]Поиск по RAG + генерация ответа...[/dim]")
        answer = chat.chat(user_input)

        _print_task_memory(chat.task_memory, answer.turn)
        _print_answer(answer)


# ─────────────────────────────────────────────────────────────────────────────
# Автодемо
# ─────────────────────────────────────────────────────────────────────────────


def _run_scenario(
    chat: RagMemoryChat,
    questions: list[str],
    title: str,
    pause: float = 0.3,
) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()

    for i, question in enumerate(questions, 1):
        console.print(
            Panel(
                f"[bold yellow]Вопрос {i}/{len(questions)}:[/bold yellow]  {question}",
                border_style="yellow",
                padding=(0, 2),
                expand=False,
            )
        )

        console.print("[dim]  → RAG-поиск...[/dim]")
        answer = chat.chat(question)

        _print_task_memory(chat.task_memory, answer.turn)
        _print_answer(answer)

        time.sleep(pause)

    console.print(Rule(f"[bold green]Сценарий завершён: {len(questions)} ходов[/bold green]"))
    _print_history_summary(chat)


def run_memory_demo(
    api_key: str,
    model: str,
    strategy: str = "structural",
    top_k: int = 5,
    pause: float = 0.3,
) -> None:
    """Автодемо: два длинных сценария (без ввода с клавиатуры).

    Сценарий 1 — «Токенизация в NLP» (11 сообщений)
    Сценарий 2 — «Fine-tuning BERT» (12 сообщений)

    Демонстрирует:
      - сохранение истории диалога между ходами
      - накопление памяти задачи (цель, уточнения, ограничения)
      - вывод источников при каждом ответе
      - непрерывность контекста (ассистент не теряет цель)
    """
    _print_header()

    agent = RagAgent(
        api_key=api_key,
        model=model,
        index_dir=_INDEX_DIR,
        strategy=strategy,
        top_k=top_k,
        rewrite_enabled=True,
    )

    console.print("[dim]Загружаем FAISS-индекс...[/dim]")
    agent._ensure_index()
    console.print(f"[dim]Индекс готов: {agent._index.index.ntotal} векторов[/dim]\n")  # type: ignore[union-attr]

    # ── Сценарий 1 ────────────────────────────────────────────────────────────
    chat1 = RagMemoryChat(agent=agent)
    _run_scenario(
        chat1,
        _SCENARIO_1,
        title="Сценарий 1: Токенизация в NLP  (11 вопросов)",
        pause=pause,
    )

    console.print()
    console.print(Rule("[bold dim]Сценарий 1 завершён. Начинаем сценарий 2...[/bold dim]"))
    console.print()
    time.sleep(1.0)

    # ── Сценарий 2 ────────────────────────────────────────────────────────────
    chat2 = RagMemoryChat(agent=agent)
    _run_scenario(
        chat2,
        _SCENARIO_2,
        title="Сценарий 2: Fine-tuning BERT для классификации  (12 вопросов)",
        pause=pause,
    )

    # ── Итоговая сводка ───────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text.assemble(
            ("Демо завершено!\n\n", "bold green"),
            ("Сценарий 1 — Токенизация:\n", "bold"),
            (f"  Ходов: {len(chat1.history)}\n", ""),
            (f"  Память задачи: цель = «{chat1.task_memory.goal or 'не определена'}»\n", "dim"),
            (f"  Уточнений: {len(chat1.task_memory.clarifications)}, "
             f"ограничений: {len(chat1.task_memory.constraints)}\n\n", "dim"),
            ("Сценарий 2 — Fine-tuning:\n", "bold"),
            (f"  Ходов: {len(chat2.history)}\n", ""),
            (f"  Память задачи: цель = «{chat2.task_memory.goal or 'не определена'}»\n", "dim"),
            (f"  Уточнений: {len(chat2.task_memory.clarifications)}, "
             f"ограничений: {len(chat2.task_memory.constraints)}\n\n", "dim"),
            ("Все ответы содержали источники из HuggingFace NLP Course.", "dim"),
        ),
        title="[bold] Результат [/bold]",
        border_style="green",
    ))
