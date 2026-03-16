"""Load and parse .mdx / .md documents from a directory."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _strip_mdx_components(text: str) -> str:
    """Remove JSX/MDX component tags, keeping inner text when possible."""
    # Remove self-closing tags like <Tip />, <Youtube .../>
    text = re.sub(r"<[A-Z][^>]*/\s*>", "", text)
    # Remove opening/closing component tags but keep content between them
    text = re.sub(r"<[A-Z][^>]*>", "", text)
    text = re.sub(r"</[A-Z][^>]*>", "", text)
    return text


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-like frontmatter enclosed in --- ... ---."""
    meta: dict = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if match:
        fm_text = match.group(1)
        for line in fm_text.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip('"').strip("'")
        text = text[match.end():]
    return meta, text


def _infer_title(text: str, fallback: str) -> str:
    """Return first H1 heading or fallback."""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Recursively load all .mdx and .md files from docs_dir."""
    docs_dir = Path(docs_dir)
    documents: list[Document] = []

    for path in sorted(docs_dir.rglob("*.md*")):
        if path.suffix not in {".md", ".mdx"}:
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = _extract_frontmatter(raw)
        body = _strip_mdx_components(body)

        rel_path = path.relative_to(docs_dir)
        parts = rel_path.parts
        chapter = parts[0] if len(parts) > 1 else "root"

        title = frontmatter.get("title") or _infer_title(body, path.stem)

        documents.append(Document(
            text=body,
            metadata={
                "source": "huggingface_nlp_course",
                "file": str(rel_path),
                "chapter": chapter,
                "title": title,
            },
        ))

    return documents
