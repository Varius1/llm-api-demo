"""Compare two chunking strategies with statistics and sample retrieval."""
from __future__ import annotations

import statistics

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chunker import Chunk
from .embedder import Embedder
from .indexer import FaissIndex

console = Console()

_TEST_QUERIES = [
    "What are Transformer models and how do they work?",
    "How to fine-tune a pretrained model on a classification task?",
    "What is tokenization and how does it work in NLP?",
]


def _chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(lengths),
        "avg_chars": round(statistics.mean(lengths), 1) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
    }


def _bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(value / max_value * width)) if max_value else 0
    return "█" * filled + "░" * (width - filled)


def _histogram(chunks: list[Chunk], bins: int = 8) -> str:
    lengths = [len(c.text) for c in chunks]
    if not lengths:
        return "(no data)"
    lo, hi = min(lengths), max(lengths)
    if lo == hi:
        return f"All chunks: {lo} chars"
    step = (hi - lo) / bins
    counts = [0] * bins
    for length in lengths:
        bucket = min(int((length - lo) / step), bins - 1)
        counts[bucket] += 1
    max_count = max(counts)
    lines = []
    for i, count in enumerate(counts):
        range_lo = int(lo + i * step)
        range_hi = int(lo + (i + 1) * step)
        bar = _bar(count, max_count, width=25)
        lines.append(f"  {range_lo:>5}-{range_hi:<5} {bar} {count}")
    return "\n".join(lines)


def print_comparison(
    fixed_chunks: list[Chunk],
    struct_chunks: list[Chunk],
    embedder: Embedder,
    fixed_index: FaissIndex,
    struct_index: FaissIndex,
) -> None:
    fixed_stats = _chunk_stats(fixed_chunks)
    struct_stats = _chunk_stats(struct_chunks)

    # --- Statistics table ---
    table = Table(title="Chunking Strategy Comparison", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Fixed (512 chars)", style="yellow")
    table.add_column("Structural (by headings)", style="green")

    rows = [
        ("Total chunks", "count"),
        ("Total chars indexed", "total_chars"),
        ("Avg chunk length", "avg_chars"),
        ("Min chunk length", "min_chars"),
        ("Max chunk length", "max_chars"),
        ("Std deviation", "stdev"),
    ]
    for label, key in rows:
        table.add_row(label, str(fixed_stats[key]), str(struct_stats[key]))

    console.print(table)

    # --- Histograms ---
    console.print(Panel(
        f"[yellow]Fixed chunking[/yellow]\n{_histogram(fixed_chunks)}\n\n"
        f"[green]Structural chunking[/green]\n{_histogram(struct_chunks)}",
        title="Chunk length distribution (chars)",
        expand=False,
    ))

    # --- Sample retrieval comparison ---
    console.print("\n[bold]Sample retrieval (top-3 per query)[/bold]\n")
    query_embeddings = embedder.encode(_TEST_QUERIES, show_progress=False)

    for i, query in enumerate(_TEST_QUERIES):
        q_emb = query_embeddings[i]
        console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", expand=False))

        for strategy_name, index in [("Fixed", fixed_index), ("Structural", struct_index)]:
            results = index.search(q_emb, top_k=3)
            result_table = Table(show_header=True, header_style="bold")
            result_table.add_column("Score", width=6)
            result_table.add_column("File", width=20)
            result_table.add_column("Section", width=25)
            result_table.add_column("Snippet", width=60)
            for chunk, score in results:
                snippet = chunk.text[:120].replace("\n", " ")
                result_table.add_row(
                    f"{score:.3f}",
                    chunk.metadata.get("file", ""),
                    chunk.metadata.get("section", "")[:25],
                    snippet,
                )
            color = "yellow" if strategy_name == "Fixed" else "green"
            console.print(f"  [{color}]{strategy_name}[/{color}]")
            console.print(result_table)
        console.print()
