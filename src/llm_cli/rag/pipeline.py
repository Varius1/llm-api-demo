"""Full RAG indexing pipeline: load → chunk → embed → index → (compare)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .chunker import Chunk, FixedChunker, StructuralChunker
from .compare import print_comparison
from .embedder import Embedder
from .indexer import FaissIndex
from .loader import load_documents

console = Console()

_PROJECT_ROOT = Path(__file__).parents[3]
_DOCS_DIR = _PROJECT_ROOT / "data" / "docs"
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"


def _build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    output_dir: Path,
    strategy_label: str,
) -> FaissIndex:
    console.print(f"\n[bold]Building [cyan]{strategy_label}[/cyan] index ({len(chunks)} chunks)...[/bold]")
    t0 = time.perf_counter()

    texts = [c.text for c in chunks]
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task(f"Embedding {len(texts)} chunks...", total=None)
        embeddings = embedder.encode(texts, show_progress=False)
        prog.update(task, completed=True)

    index = FaissIndex(dim=embedder.dim)
    index.add(chunks, embeddings)
    index.save(output_dir)

    elapsed = time.perf_counter() - t0
    console.print(
        f"  [green]✓[/green] Saved to [dim]{output_dir}[/dim] "
        f"([yellow]{len(chunks)}[/yellow] chunks, {elapsed:.1f}s)"
    )
    return index


def run_pipeline(
    docs_dir: Path = _DOCS_DIR,
    index_dir: Path = _INDEX_DIR,
    strategy: str = "both",
    compare: bool = True,
    chunk_size: int = 512,
    overlap: int = 64,
) -> None:
    # 1. Load documents
    console.print(f"\n[bold blue]Loading documents from[/bold blue] [dim]{docs_dir}[/dim]")
    docs = load_documents(docs_dir)
    if not docs:
        console.print(f"[red]No documents found in {docs_dir}[/red]")
        return
    total_chars = sum(len(d.text) for d in docs)
    console.print(
        f"  Loaded [yellow]{len(docs)}[/yellow] documents, "
        f"[yellow]{total_chars:,}[/yellow] chars total"
    )

    fixed_index = struct_index = None

    # 2a. Fixed chunking
    if strategy in ("fixed", "both"):
        chunker = FixedChunker(chunk_size=chunk_size, overlap=overlap)
        fixed_chunks: list[Chunk] = []
        for doc in docs:
            fixed_chunks.extend(chunker.chunk(doc))
        embedder = Embedder()
        fixed_index = _build_index(fixed_chunks, embedder, index_dir / "fixed", "fixed")

    # 2b. Structural chunking
    if strategy in ("structural", "both"):
        chunker_struct = StructuralChunker()
        struct_chunks: list[Chunk] = []
        for doc in docs:
            struct_chunks.extend(chunker_struct.chunk(doc))
        embedder = embedder if strategy == "both" else Embedder()  # type: ignore[has-type]
        struct_index = _build_index(struct_chunks, embedder, index_dir / "structural", "structural")

    # 3. Compare strategies
    if compare and strategy == "both" and fixed_index and struct_index:
        console.print("\n[bold blue]═══ Strategy Comparison ═══[/bold blue]")
        print_comparison(fixed_chunks, struct_chunks, embedder, fixed_index, struct_index)  # type: ignore[arg-type]

    console.print("\n[bold green]Pipeline complete.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Indexing Pipeline")
    parser.add_argument("--docs-dir", type=Path, default=_DOCS_DIR, help="Directory with source documents")
    parser.add_argument("--index-dir", type=Path, default=_INDEX_DIR, help="Output directory for indexes")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structural", "both"],
        default="both",
        help="Chunking strategy to use",
    )
    parser.add_argument("--no-compare", action="store_true", help="Skip strategy comparison")
    parser.add_argument("--chunk-size", type=int, default=512, help="Fixed chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap for fixed chunking")
    args = parser.parse_args()

    run_pipeline(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        strategy=args.strategy,
        compare=not args.no_compare,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
