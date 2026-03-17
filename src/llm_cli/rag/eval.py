"""10 контрольных вопросов по HuggingFace NLP Course + функция оценки RAG vs no-RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    from .rag_agent import RagAgent

console = Console()

# ---------------------------------------------------------------------------
# 10 контрольных вопросов
# ---------------------------------------------------------------------------

EVAL_QUESTIONS: list[dict] = [
    {
        "id": 1,
        "question": "Что такое токенизатор и зачем он нужен в NLP?",
        "expected_keywords": ["токен", "token", "subword", "tokenizer", "текст", "text"],
        "expected_sources": ["chapter2"],
        "hint": "Базовое понятие — как текст превращается в числа для модели",
    },
    {
        "id": 2,
        "question": "Чем отличается WordPiece от BPE (Byte Pair Encoding)?",
        "expected_keywords": ["wordpiece", "bpe", "byte pair", "vocabulary", "merge", "слияние"],
        "expected_sources": ["chapter2", "chapter6"],
        "hint": "Два алгоритма токенизации, используемых в BERT и GPT",
    },
    {
        "id": 3,
        "question": "Что такое attention mask и когда он используется при батчевой обработке?",
        "expected_keywords": ["attention", "mask", "padding", "batch", "ignore"],
        "expected_sources": ["chapter2"],
        "hint": "Механизм игнорирования паддинг-токенов в трансформерах",
    },
    {
        "id": 4,
        "question": "Как работает padding и truncation при токенизации последовательностей разной длины?",
        "expected_keywords": ["padding", "truncation", "max_length", "pad_token", "sequence"],
        "expected_sources": ["chapter2"],
        "hint": "Выравнивание длин в батче — обрезание и дополнение",
    },
    {
        "id": 5,
        "question": "Что такое fine-tuning и чем он отличается от pre-training предобученной модели?",
        "expected_keywords": ["fine-tuning", "fine-tune", "pre-train", "pretrain", "задача", "task"],
        "expected_sources": ["chapter3"],
        "hint": "Как переиспользуют предобученные трансформеры для конкретных задач",
    },
    {
        "id": 6,
        "question": "Как загрузить датасет с помощью библиотеки HuggingFace Datasets?",
        "expected_keywords": ["load_dataset", "dataset", "split", "train", "hub"],
        "expected_sources": ["chapter3", "chapter5"],
        "hint": "API для работы с датасетами из Hub и локальных файлов",
    },
    {
        "id": 7,
        "question": "Как работает Trainer API в библиотеке transformers для обучения модели?",
        "expected_keywords": ["trainer", "trainingarguments", "training_args", "evaluate", "train"],
        "expected_sources": ["chapter3"],
        "hint": "Высокоуровневый API для тренировки — настройка, запуск, метрики",
    },
    {
        "id": 8,
        "question": "Что такое BERT и в чём его архитектурные особенности по сравнению с GPT?",
        "expected_keywords": ["bert", "encoder", "bidirectional", "masked", "mlm"],
        "expected_sources": ["chapter1", "chapter2"],
        "hint": "Энкодерная модель с двунаправленным вниманием и предобучением на MLM",
    },
    {
        "id": 9,
        "question": "Что делает AutoModel.from_pretrained() и какие классы AutoModel существуют?",
        "expected_keywords": ["automodel", "from_pretrained", "config", "architecture", "checkpoint"],
        "expected_sources": ["chapter2", "chapter4"],
        "hint": "Автоматическое определение архитектуры из названия модели",
    },
    {
        "id": 10,
        "question": "Как сохранить и загрузить модель и токенизатор локально с помощью save_pretrained?",
        "expected_keywords": ["save_pretrained", "from_pretrained", "local", "directory", "weights"],
        "expected_sources": ["chapter2", "chapter4"],
        "hint": "Сохранение весов и конфигурации для офлайн-использования",
    },
]


# ---------------------------------------------------------------------------
# Результат одного вопроса
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    question_id: int
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]

    answer_no_rag: str = ""
    answer_rag: str = ""
    rag_sources: list[str] = field(default_factory=list)

    keywords_hit_no_rag: int = 0
    keywords_hit_rag: int = 0

    @property
    def rag_wins(self) -> bool:
        return self.keywords_hit_rag > self.keywords_hit_no_rag

    @property
    def rag_ties(self) -> bool:
        return self.keywords_hit_rag == self.keywords_hit_no_rag

    @property
    def no_rag_wins(self) -> bool:
        return self.keywords_hit_no_rag > self.keywords_hit_rag

    def _count_hits(self, answer: str, keywords: list[str]) -> int:
        lower = answer.lower()
        return sum(1 for kw in keywords if kw.lower() in lower)


def _short(text: str, max_len: int = 120) -> str:
    text = text.strip().replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text


# ---------------------------------------------------------------------------
# Основная функция eval
# ---------------------------------------------------------------------------

def run_eval(rag_agent: RagAgent) -> list[EvalResult]:
    """Прогнать 10 контрольных вопросов, вывести Rich-таблицу сравнения.

    Args:
        rag_agent: Инициализированный RagAgent.

    Returns:
        Список EvalResult для каждого вопроса.
    """
    results: list[EvalResult] = []

    console.print()
    console.print(Panel(
        "[bold cyan]RAG Evaluation: 10 контрольных вопросов[/bold cyan]\n"
        "[dim]HuggingFace NLP Course — сравнение ответов без RAG и с RAG[/dim]",
        border_style="cyan",
    ))
    console.print()

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        console.rule(f"[bold]Вопрос {i}/10[/bold]")
        console.print(f"[bold yellow]{q['question']}[/bold yellow]")
        console.print(f"[dim]Подсказка: {q['hint']}[/dim]\n")

        # --- Ответ без RAG ---
        console.print("[bold red]⟳[/bold red] Запрос без RAG...")
        ans_no_rag = rag_agent.ask(q["question"], use_rag=False)

        # --- Ответ с RAG ---
        console.print("[bold green]⟳[/bold green] Запрос с RAG...")
        ans_rag = rag_agent.ask(q["question"], use_rag=True)

        result = EvalResult(
            question_id=q["id"],
            question=q["question"],
            expected_keywords=q["expected_keywords"],
            expected_sources=q["expected_sources"],
            answer_no_rag=ans_no_rag.text,
            answer_rag=ans_rag.text,
            rag_sources=ans_rag.sources,
        )
        result.keywords_hit_no_rag = result._count_hits(ans_no_rag.text, q["expected_keywords"])
        result.keywords_hit_rag = result._count_hits(ans_rag.text, q["expected_keywords"])

        # Показываем результат вопроса прямо сейчас
        _print_question_result(result)
        results.append(result)
        console.print()

    # Итоговая таблица
    _print_summary_table(results)
    return results


def _print_question_result(r: EvalResult) -> None:
    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold white", expand=True)
    tbl.add_column("Режим", style="bold", width=12)
    tbl.add_column("Ответ (кратко)", ratio=3)
    tbl.add_column("Ключевые слова", justify="center", width=16)

    kw_total = len(r.expected_keywords)

    tbl.add_row(
        "[red]БЕЗ RAG[/red]",
        _short(r.answer_no_rag),
        f"{r.keywords_hit_no_rag}/{kw_total}",
    )

    sources_str = ", ".join(r.rag_sources[:3]) if r.rag_sources else "—"
    tbl.add_row(
        "[green]С RAG[/green]",
        _short(r.answer_rag),
        f"[bold]{r.keywords_hit_rag}/{kw_total}[/bold]",
    )

    console.print(tbl)
    console.print(f"  [dim]Источники RAG: {sources_str}[/dim]")
    if r.rag_wins:
        verdict = "[bold green]✓ RAG выиграл[/bold green]"
    elif r.rag_ties:
        verdict = "[bold yellow]≈ Ничья[/bold yellow]"
    else:
        verdict = "[bold red]✗ Без RAG лучше[/bold red]"
    console.print(f"  {verdict}")


def _print_summary_table(results: list[EvalResult]) -> None:
    console.print()
    console.rule("[bold cyan]Итоги оценки[/bold cyan]")
    console.print()

    tbl = Table(title="Сводная таблица: 10 вопросов", box=box.HEAVY_HEAD, expand=True)
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("Вопрос", ratio=3)
    tbl.add_column("Без RAG", justify="center", width=10)
    tbl.add_column("С RAG", justify="center", width=10)
    tbl.add_column("Источники", ratio=2)
    tbl.add_column("Итог", justify="center", width=14)

    rag_wins_count = 0
    ties_count = 0
    no_rag_wins_count = 0

    for r in results:
        kw_total = len(r.expected_keywords)
        if r.rag_wins:
            verdict_markup = "[green]✓ RAG[/green]"
            rag_wins_count += 1
        elif r.rag_ties:
            verdict_markup = "[yellow]≈ Ничья[/yellow]"
            ties_count += 1
        else:
            verdict_markup = "[red]✗ no-RAG[/red]"
            no_rag_wins_count += 1

        tbl.add_row(
            str(r.question_id),
            _short(r.question, max_len=60),
            f"{r.keywords_hit_no_rag}/{kw_total}",
            f"[bold]{r.keywords_hit_rag}/{kw_total}[/bold]",
            ", ".join(r.rag_sources[:2]) or "—",
            verdict_markup,
        )

    console.print(tbl)
    console.print()
    console.print(
        f"[green]✓ RAG выиграл[/green]: [cyan]{rag_wins_count}[/cyan]  "
        f"[yellow]≈ Ничья[/yellow]: [cyan]{ties_count}[/cyan]  "
        f"[red]✗ no-RAG лучше[/red]: [cyan]{no_rag_wins_count}[/cyan]"
        f"  [dim](по ключевым словам в ответе)[/dim]"
    )
    console.print()

    if rag_wins_count >= 7:
        console.print("[bold green]Вывод: RAG значительно улучшает качество ответов[/bold green]")
    elif rag_wins_count >= 4:
        console.print("[bold yellow]Вывод: RAG улучшает ответы в большинстве случаев[/bold yellow]")
    elif rag_wins_count + ties_count >= 7:
        console.print("[bold yellow]Вывод: RAG не хуже, в ряде случаев добавляет точности из источников[/bold yellow]")
    else:
        console.print(
            "[bold red]Вывод: RAG уступает по счётчику ключевых слов — "
            "возможно, модель без RAG даёт более развёрнутые ответы на русском[/bold red]"
        )
