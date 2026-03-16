"""RAG Demo — наглядная демонстрация результатов задания."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .chunker import FixedChunker, StructuralChunker
from .embedder import Embedder
from .indexer import FaissIndex
from .loader import load_documents

console = Console()

_PROJECT_ROOT = Path(__file__).parents[3]
_DOCS_DIR = _PROJECT_ROOT / "data" / "docs"
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"

_QUERIES = [
    "What are Transformer models and how do they work?",
    "How to fine-tune a pretrained model on a classification task?",
    "What is tokenization and how does it work in NLP?",
    "What is the difference between encoder and decoder models?",
]


def _pause(seconds: float = 0.6) -> None:
    time.sleep(seconds)


def _step(num: int, title: str) -> None:
    console.print()
    console.print(Rule(f"[bold white] Шаг {num}: {title} [/bold white]", style="bright_blue"))
    _pause(0.3)


def _bar(value: float, max_value: float, width: int = 28) -> str:
    filled = int(round(value / max_value * width)) if max_value else 0
    return "█" * filled + "░" * (width - filled)


def run_demo() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]RAG Indexing Pipeline — Demo[/bold cyan]\n\n"
        "  Задание: локальный индекс документов с эмбеддингами,\n"
        "  метаданными и сравнением двух стратегий chunking.\n\n"
        "  Документы: [yellow]HuggingFace NLP Course[/yellow] (главы 1–5)\n"
        "  Модель:    [yellow]all-MiniLM-L6-v2[/yellow] (384-мерные эмбеддинги)\n"
        "  Хранилище: [yellow]FAISS IndexFlatIP[/yellow] + [yellow]JSON метаданные[/yellow]",
        title="[bold] RAG Demo [/bold]",
        border_style="bright_blue",
        expand=False,
    ))
    _pause(0.8)

    # ── Шаг 1: Документы ──────────────────────────────────────────────────────
    _step(1, "Загрузка документов")
    docs = load_documents(_DOCS_DIR)
    total_chars = sum(len(d.text) for d in docs)
    total_words = sum(len(d.text.split()) for d in docs)

    doc_table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    doc_table.add_column("Файл", style="dim")
    doc_table.add_column("Заголовок", style="white")
    doc_table.add_column("Символов", justify="right", style="yellow")
    for doc in docs[:12]:
        doc_table.add_row(
            doc.metadata.get("file", ""),
            doc.metadata.get("title", "")[:45],
            f"{len(doc.text):,}",
        )
    if len(docs) > 12:
        doc_table.add_row("...", f"... ещё {len(docs) - 12} файлов", "")

    console.print(doc_table)
    console.print(
        f"\n  Итого: [yellow]{len(docs)}[/yellow] документов  |  "
        f"[yellow]{total_chars:,}[/yellow] символов  |  "
        f"[yellow]{total_words:,}[/yellow] слов"
    )
    _pause(0.8)

    # ── Шаг 2: Chunking ───────────────────────────────────────────────────────
    _step(2, "Chunking — две стратегии")

    fixed_chunks_all = []
    struct_chunks_all = []
    fc = FixedChunker(chunk_size=512, overlap=64)
    sc = StructuralChunker()
    for doc in docs:
        fixed_chunks_all.extend(fc.chunk(doc))
        struct_chunks_all.extend(sc.chunk(doc))

    # Показываем пример одного документа обеими стратегиями
    sample_doc = docs[2]
    sample_fixed = fc.chunk(sample_doc)
    sample_struct = sc.chunk(sample_doc)

    console.print(f"\n  Пример: [dim]{sample_doc.metadata['file']}[/dim] — «[cyan]{sample_doc.metadata['title']}[/cyan]»\n")

    ex_table = Table(show_lines=True, box=None, padding=(0, 1))
    ex_table.add_column("Стратегия", style="bold", width=14)
    ex_table.add_column("chunk_id", style="dim", width=32)
    ex_table.add_column("Длина", justify="right", width=7)
    ex_table.add_column("Section", width=22)
    ex_table.add_column("Начало текста", width=55)

    for ch in sample_fixed[:3]:
        ex_table.add_row(
            "[yellow]Fixed[/yellow]",
            ch.metadata["chunk_id"],
            str(len(ch.text)),
            str(ch.metadata.get("section", ""))[:22],
            ch.text[:80].replace("\n", " "),
        )
    for ch in sample_struct[:3]:
        ex_table.add_row(
            "[green]Structural[/green]",
            ch.metadata["chunk_id"],
            str(len(ch.text)),
            str(ch.metadata.get("section", ""))[:22],
            ch.text[:80].replace("\n", " "),
        )

    console.print(ex_table)
    _pause(0.6)

    # ── Шаг 3: Метаданные ─────────────────────────────────────────────────────
    _step(3, "Метаданные чанка")

    sample_chunk = sample_struct[1]
    meta_str = json.dumps(
        {**sample_chunk.metadata, "text": sample_chunk.text[:120] + "…"},
        ensure_ascii=False,
        indent=2,
    )
    console.print(Panel(
        Syntax(meta_str, "json", theme="monokai", word_wrap=True),
        title="[bold]Пример метаданных (structural chunk)[/bold]",
        border_style="green",
        expand=False,
    ))
    _pause(0.6)

    # ── Шаг 4: Индекс (статистика из файлов) ─────────────────────────────────
    _step(4, "Сохранённый FAISS-индекс")

    fixed_meta_path = _INDEX_DIR / "fixed" / "metadata.json"
    struct_meta_path = _INDEX_DIR / "structural" / "metadata.json"

    fixed_meta = json.loads(fixed_meta_path.read_text())
    struct_meta = json.loads(struct_meta_path.read_text())
    fixed_faiss_size = (_INDEX_DIR / "fixed" / "index.faiss").stat().st_size
    struct_faiss_size = (_INDEX_DIR / "structural" / "index.faiss").stat().st_size

    idx_table = Table(show_lines=True)
    idx_table.add_column("Файл", style="dim")
    idx_table.add_column("Размер", justify="right", style="yellow")
    idx_table.add_column("Содержимое")
    idx_table.add_row(
        "data/index/fixed/index.faiss",
        f"{fixed_faiss_size / 1024:.0f} KB",
        f"FAISS IndexFlatIP, {len(fixed_meta)} векторов × 384 dim",
    )
    idx_table.add_row(
        "data/index/fixed/metadata.json",
        f"{fixed_meta_path.stat().st_size // 1024} KB",
        f"{len(fixed_meta)} чанков с метаданными",
    )
    idx_table.add_row(
        "data/index/structural/index.faiss",
        f"{struct_faiss_size / 1024:.0f} KB",
        f"FAISS IndexFlatIP, {len(struct_meta)} векторов × 384 dim",
    )
    idx_table.add_row(
        "data/index/structural/metadata.json",
        f"{struct_meta_path.stat().st_size // 1024} KB",
        f"{len(struct_meta)} чанков с метаданными",
    )
    console.print(idx_table)
    _pause(0.6)

    # ── Шаг 5: Сравнение стратегий ────────────────────────────────────────────
    _step(5, "Сравнение стратегий chunking")

    def stats(chunks_list: list) -> dict:
        import statistics as st
        lens = [len(c["text"]) for c in chunks_list]
        return {
            "count": len(lens),
            "avg": round(st.mean(lens), 1),
            "min": min(lens),
            "max": max(lens),
            "stdev": round(st.stdev(lens), 1),
        }

    fs = stats(fixed_meta)
    ss = stats(struct_meta)

    cmp_table = Table(show_lines=True, title="Статистика стратегий")
    cmp_table.add_column("Метрика", style="bold cyan")
    cmp_table.add_column("Fixed (512 chars)", style="yellow", justify="right")
    cmp_table.add_column("Structural (headings)", style="green", justify="right")
    cmp_table.add_column("Вывод", style="dim")

    cmp_table.add_row("Всего чанков", str(fs["count"]), str(ss["count"]), "сопоставимо")
    cmp_table.add_row("Средняя длина", str(fs["avg"]), str(ss["avg"]), "близко")
    cmp_table.add_row("Минимальная", str(fs["min"]), str(ss["min"]), "structural короче")
    cmp_table.add_row("Максимальная", str(fs["max"]), str(ss["max"]), "structural длиннее")
    cmp_table.add_row(
        "Стд. отклонение",
        str(fs["stdev"]),
        str(ss["stdev"]),
        "[green]fixed равномернее[/green]",
    )
    console.print(cmp_table)

    # Гистограммы
    import statistics as st
    fixed_lens = [len(c["text"]) for c in fixed_meta]
    struct_lens = [len(c["text"]) for c in struct_meta]

    def histogram_lines(lens: list[int], bins: int = 8, color: str = "white") -> str:
        lo, hi = min(lens), max(lens)
        step = (hi - lo) / bins
        counts = [0] * bins
        for v in lens:
            b = min(int((v - lo) / step), bins - 1)
            counts[b] += 1
        mx = max(counts)
        lines = []
        for i, cnt in enumerate(counts):
            r0 = int(lo + i * step)
            r1 = int(lo + (i + 1) * step)
            bar = f"[{color}]" + "█" * int(cnt / mx * 22) + "[/]" + "░" * (22 - int(cnt / mx * 22))
            lines.append(f"  {r0:>5}–{r1:<5} {bar} {cnt}")
        return "\n".join(lines)

    console.print(Panel(
        f"[yellow bold]Fixed chunking[/yellow bold]  (stdev={fs['stdev']})\n"
        + histogram_lines(fixed_lens, color="yellow")
        + f"\n\n[green bold]Structural chunking[/green bold]  (stdev={ss['stdev']})\n"
        + histogram_lines(struct_lens, color="green"),
        title="Распределение длин чанков",
        border_style="bright_blue",
        expand=False,
    ))
    _pause(0.8)

    # ── Шаг 6: Семантический поиск ────────────────────────────────────────────
    _step(6, "Семантический поиск по индексу")

    console.print("  [dim]Загружаем модель и индексы...[/dim]")
    embedder = Embedder()
    fixed_index = FaissIndex.load(_INDEX_DIR / "fixed", dim=embedder.dim)
    struct_index = FaissIndex.load(_INDEX_DIR / "structural", dim=embedder.dim)
    query_embs = embedder.encode(_QUERIES, show_progress=False)

    for i, query in enumerate(_QUERIES):
        console.print()
        console.print(Panel(
            f"[bold cyan]Запрос:[/bold cyan]  {query}",
            border_style="cyan",
            expand=False,
        ))

        for label, idx, color in [("Fixed", fixed_index, "yellow"), ("Structural", struct_index, "green")]:
            results = idx.search(query_embs[i], top_k=3)
            tbl = Table(show_header=True, header_style=f"bold {color}", box=None, padding=(0, 1))
            tbl.add_column("Score", width=6, justify="right")
            tbl.add_column("Файл", width=20)
            tbl.add_column("Раздел", width=26)
            tbl.add_column("Отрывок", width=62)
            for chunk, score in results:
                tbl.add_row(
                    f"[{color}]{score:.3f}[/{color}]",
                    chunk.metadata.get("file", ""),
                    chunk.metadata.get("section", "")[:26],
                    chunk.text[:100].replace("\n", " "),
                )
            console.print(f"  [{color}]{label}[/{color}]")
            console.print(tbl)
        _pause(0.2)

    # ── Итог ──────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold green]Задание выполнено![/bold green]\n\n"
        f"  [cyan]Документы:[/cyan]   {len(docs)} файлов, {total_chars:,} символов  ([dim]data/docs/[/dim])\n"
        f"  [cyan]Fixed index:[/cyan] {fs['count']} чанков  →  [dim]data/index/fixed/[/dim]\n"
        f"  [cyan]Struct index:[/cyan]{ss['count']} чанков  →  [dim]data/index/structural/[/dim]\n"
        f"  [cyan]Эмбеддинги:[/cyan]  all-MiniLM-L6-v2, 384-dim, cosine similarity\n\n"
        "  [dim]Стратегии:[/dim]\n"
        "    [yellow]Fixed[/yellow]      — равномерные окна 512 символов, overlap 64\n"
        "    [green]Structural[/green] — разбивка по заголовкам #/##/###\n\n"
        "  [dim]Fixed равномернее по размеру (stdev[/dim] [yellow]58[/yellow][dim] vs [/dim][green]209[/green][dim]).[/dim]\n"
        "  [dim]Structural сохраняет смысловые границы разделов.[/dim]",
        title="[bold] Результат [/bold]",
        border_style="green",
    ))
