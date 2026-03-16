"""Two chunking strategies: FixedChunker and StructuralChunker."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import Document


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking
# ---------------------------------------------------------------------------

class FixedChunker:
    """Split text into overlapping fixed-size character windows."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        chunk_num = 0

        while idx < len(text):
            end = min(idx + self.chunk_size, len(text))
            snippet = text[idx:end].strip()
            if snippet:
                file_key = doc.metadata.get("file", "unknown").replace("/", "_").replace(".", "_")
                chunks.append(Chunk(
                    text=snippet,
                    metadata={
                        **doc.metadata,
                        "strategy": "fixed",
                        "chunk_id": f"{file_key}_fixed_{chunk_num:04d}",
                        "chunk_num": chunk_num,
                        "char_start": idx,
                        "char_end": end,
                        "section": doc.metadata.get("title", ""),
                    },
                ))
                chunk_num += 1
            idx += step

        return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Structural chunking (by Markdown headings)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[int, str, str]]:
    """
    Return list of (heading_level, heading_text, section_body).
    heading_level=0 means content before first heading.
    """
    sections: list[tuple[int, str, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [(0, "", text)]

    # Content before first heading
    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append((0, "preamble", pre))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body or heading:
            sections.append((level, heading, body))

    return sections


class StructuralChunker:
    """
    Split documents by Markdown headings (H1/H2/H3).
    Sections that exceed max_section_chars are further split with a fixed window.
    """

    def __init__(self, max_section_chars: int = 1500, overflow_chunk_size: int = 512, overflow_overlap: int = 64):
        self.max_section_chars = max_section_chars
        self.overflow_chunk_size = overflow_chunk_size
        self.overflow_overlap = overflow_overlap

    def _overflow_split(self, text: str) -> list[str]:
        step = self.overflow_chunk_size - self.overflow_overlap
        parts = []
        idx = 0
        while idx < len(text):
            parts.append(text[idx:idx + self.overflow_chunk_size].strip())
            idx += step
        return [p for p in parts if p]

    def chunk(self, doc: Document) -> list[Chunk]:
        sections = _split_by_headings(doc.text)
        chunks: list[Chunk] = []
        chunk_num = 0
        file_key = doc.metadata.get("file", "unknown").replace("/", "_").replace(".", "_")

        for level, heading, body in sections:
            combined = (f"{'#' * level} {heading}\n\n{body}".strip() if heading else body)
            if not combined:
                continue

            texts_to_add: list[tuple[str, str]] = []
            if len(combined) > self.max_section_chars:
                for part in self._overflow_split(combined):
                    texts_to_add.append((part, heading))
            else:
                texts_to_add.append((combined, heading))

            for snippet, section_title in texts_to_add:
                chunks.append(Chunk(
                    text=snippet,
                    metadata={
                        **doc.metadata,
                        "strategy": "structural",
                        "chunk_id": f"{file_key}_struct_{chunk_num:04d}",
                        "chunk_num": chunk_num,
                        "section": section_title or doc.metadata.get("title", ""),
                        "heading_level": level,
                    },
                ))
                chunk_num += 1

        return chunks
