"""Post-retrieval relevance filtering and heuristic reranking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .chunker import Chunk

PostRetrievalMode = Literal["off", "threshold", "rerank"]

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


@dataclass
class PostRetrievalStats:
    mode: PostRetrievalMode
    raw_count: int
    filtered_count: int
    selected_count: int
    avg_similarity: float
    min_similarity: float
    top_k_before: int
    top_k_after: int
    fallback_used: bool = False
    query_rewritten: bool = False
    rewritten_query: str | None = None


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text) if len(m.group(0)) >= 3}


def _lexical_overlap(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize(text)
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _normalize_similarity(score: float) -> float:
    # For normalized embeddings + inner product this is close to cosine in [-1, 1].
    value = (score + 1.0) / 2.0
    return max(0.0, min(1.0, value))


def apply_post_retrieval(
    chunks: list[tuple[Chunk, float]],
    *,
    query: str,
    mode: PostRetrievalMode,
    top_k_before: int,
    top_k_after: int,
    min_similarity: float,
) -> tuple[list[tuple[Chunk, float]], PostRetrievalStats]:
    """Apply optional post-retrieval step and return selected chunks + stats."""
    if not chunks:
        return [], PostRetrievalStats(
            mode=mode,
            raw_count=0,
            filtered_count=0,
            selected_count=0,
            avg_similarity=0.0,
            min_similarity=min_similarity,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            fallback_used=False,
        )

    raw_count = len(chunks)

    if mode == "off":
        selected = chunks[:top_k_after]
        avg_similarity = sum(score for _, score in selected) / len(selected) if selected else 0.0
        return selected, PostRetrievalStats(
            mode=mode,
            raw_count=raw_count,
            filtered_count=raw_count,
            selected_count=len(selected),
            avg_similarity=avg_similarity,
            min_similarity=min_similarity,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            fallback_used=False,
        )

    thresholded = [(chunk, score) for chunk, score in chunks if score >= min_similarity]

    if mode == "threshold":
        filtered_count = len(thresholded)
        selected = thresholded[:top_k_after]
        fallback_used = False
        if not selected:
            selected = chunks[:top_k_after]
            fallback_used = True
        avg_similarity = sum(score for _, score in selected) / len(selected) if selected else 0.0
        return selected, PostRetrievalStats(
            mode=mode,
            raw_count=raw_count,
            filtered_count=filtered_count,
            selected_count=len(selected),
            avg_similarity=avg_similarity,
            min_similarity=min_similarity,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            fallback_used=fallback_used,
        )

    # mode == "rerank": threshold first, then rerank by similarity + lexical overlap.
    candidates = thresholded or chunks
    fallback_used = len(thresholded) == 0

    scored: list[tuple[Chunk, float, float]] = []
    for chunk, score in candidates:
        overlap = _lexical_overlap(query, chunk.text)
        rerank_score = (0.8 * _normalize_similarity(score)) + (0.2 * overlap)
        scored.append((chunk, score, rerank_score))

    scored.sort(key=lambda item: item[2], reverse=True)
    selected = [(chunk, similarity) for chunk, similarity, _ in scored[:top_k_after]]
    avg_similarity = sum(score for _, score in selected) / len(selected) if selected else 0.0
    return selected, PostRetrievalStats(
        mode=mode,
        raw_count=raw_count,
        filtered_count=len(candidates),
        selected_count=len(selected),
        avg_similarity=avg_similarity,
        min_similarity=min_similarity,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        fallback_used=fallback_used,
    )
