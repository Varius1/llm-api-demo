"""MCP-сервер для работы с файлами проекта (stdio-транспорт).

Инструменты:
- read_file       — читает содержимое файла
- list_files      — список файлов по паттерну glob
- grep_in_files   — поиск строки/regex по нескольким файлам
- write_file      — создаёт/перезаписывает файл (только внутри проекта)
- get_file_diff   — показывает unified diff между текущим и новым содержимым
- get_project_structure — дерево каталогов
- get_git_log     — последние N коммитов git log
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-files")

# Корень проекта — два уровня вверх от этого файла (src/llm_cli → src → project_root)
_PROJECT_ROOT = Path(__file__).parents[2].resolve()

# Каталоги, которые пропускаем при обходе (всегда)
_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".tox", "dist", "build",
    "UNKNOWN.egg-info", "llm_cli.egg-info",
}

# Максимальный размер файла для чтения (байт) — защита от бинарных/огромных файлов
_MAX_FILE_SIZE = 256 * 1024  # 256 КБ


def _resolve_safe(path: str) -> Path | None:
    """Разрешить путь относительно корня проекта. Вернуть None если путь выходит за пределы."""
    if os.path.isabs(path):
        resolved = Path(path).resolve()
    else:
        resolved = (_PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
        return resolved
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# read_file
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def read_file(path: str) -> str:
    """Читает содержимое файла проекта.

    path — путь к файлу относительно корня проекта (например: src/llm_cli/agent.py)
           или абсолютный путь внутри проекта.
    Возвращает содержимое файла с номерами строк.
    """
    resolved = _resolve_safe(path)
    if resolved is None:
        return f"Ошибка: путь '{path}' выходит за пределы проекта."
    if not resolved.exists():
        return f"Ошибка: файл '{path}' не существует."
    if not resolved.is_file():
        return f"Ошибка: '{path}' — не файл (это каталог)."

    size = resolved.stat().st_size
    if size > _MAX_FILE_SIZE:
        return (
            f"Файл слишком большой ({size / 1024:.1f} КБ > {_MAX_FILE_SIZE // 1024} КБ). "
            f"Используйте grep_in_files для поиска или укажите диапазон строк."
        )

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Ошибка чтения файла '{path}': {e}"

    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
    rel = resolved.relative_to(_PROJECT_ROOT)
    return f"=== {rel} ({len(lines)} строк) ===\n{numbered}"


# ─────────────────────────────────────────────────────────────────────────────
# list_files
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Возвращает список файлов в каталоге, соответствующих паттерну.

    directory — каталог относительно корня проекта (по умолчанию: '.', т.е. весь проект).
    pattern   — glob-паттерн для имён файлов, например: '*.py', '*.md', 'test_*.py'
                Поддерживает рекурсивный поиск — паттерн применяется по всему дереву.
    Возвращает список путей относительно корня проекта.
    """
    resolved_dir = _resolve_safe(directory)
    if resolved_dir is None:
        return f"Ошибка: каталог '{directory}' выходит за пределы проекта."
    if not resolved_dir.exists():
        return f"Ошибка: каталог '{directory}' не существует."
    if not resolved_dir.is_dir():
        return f"Ошибка: '{directory}' — не каталог."

    matched: list[str] = []
    for root, dirs, files in os.walk(resolved_dir):
        # Фильтруем нежелательные каталоги на месте
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for fname in sorted(files):
            if fnmatch.fnmatch(fname, pattern):
                full = Path(root) / fname
                try:
                    rel = full.relative_to(_PROJECT_ROOT)
                    matched.append(str(rel))
                except ValueError:
                    pass

    if not matched:
        return f"Файлы по паттерну '{pattern}' в '{directory}' не найдены."

    lines = [f"Найдено файлов: {len(matched)} (паттерн: '{pattern}', каталог: '{directory}')\n"]
    lines.extend(matched)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# grep_in_files
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def grep_in_files(
    pattern: str,
    directory: str = ".",
    file_glob: str = "*.py",
    context_lines: int = 2,
    max_matches: int = 50,
) -> str:
    """Ищет строку или регулярное выражение во всех файлах, совпадающих с file_glob.

    pattern       — строка или regex для поиска (Python re-синтаксис).
    directory     — каталог поиска относительно корня проекта (по умолчанию: '.').
    file_glob     — паттерн для файлов, например: '*.py', '*.md', '*.json'.
    context_lines — количество строк контекста вокруг каждого совпадения (по умолчанию: 2).
    max_matches   — максимальное число совпадений (по умолчанию: 50).
    Возвращает файлы, строки с совпадениями и контекст.
    """
    resolved_dir = _resolve_safe(directory)
    if resolved_dir is None:
        return f"Ошибка: каталог '{directory}' выходит за пределы проекта."
    if not resolved_dir.exists():
        return f"Ошибка: каталог '{directory}' не существует."

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Ошибка: некорректный regex '{pattern}': {e}"

    results: list[str] = []
    total_matches = 0
    files_with_matches = 0

    for root, dirs, files in os.walk(resolved_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for fname in sorted(files):
            if not fnmatch.fnmatch(fname, file_glob):
                continue
            full = Path(root) / fname
            if full.stat().st_size > _MAX_FILE_SIZE:
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = text.splitlines()
            match_lines = [i for i, ln in enumerate(lines) if regex.search(ln)]
            if not match_lines:
                continue

            files_with_matches += 1
            try:
                rel = str(full.relative_to(_PROJECT_ROOT))
            except ValueError:
                rel = str(full)

            results.append(f"\n📄 {rel}  ({len(match_lines)} совпадений)")
            shown: set[int] = set()
            for mi in match_lines:
                if total_matches >= max_matches:
                    break
                start = max(0, mi - context_lines)
                end = min(len(lines) - 1, mi + context_lines)
                block_lines = list(range(start, end + 1))
                new_in_block = [i for i in block_lines if i not in shown]
                if not new_in_block:
                    continue
                if shown and min(new_in_block) > max(shown) + 1:
                    results.append("   ···")
                for i in block_lines:
                    if i in shown:
                        continue
                    shown.add(i)
                    marker = "→" if i == mi else " "
                    results.append(f"  {marker} {i + 1:4d} | {lines[i]}")
                total_matches += 1

            if total_matches >= max_matches:
                break
        if total_matches >= max_matches:
            break

    if not results:
        return f"Совпадений по паттерну '{pattern}' в файлах '{file_glob}' не найдено."

    header = (
        f"Поиск: '{pattern}'  |  файлы: '{file_glob}'  |  каталог: '{directory}'\n"
        f"Найдено: {total_matches} совпадений в {files_with_matches} файлах"
        + (f"  (показаны первые {max_matches})" if total_matches >= max_matches else "")
    )
    return header + "\n" + "\n".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# write_file
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def write_file(path: str, content: str, create_dirs: bool = True) -> str:
    """Создаёт или перезаписывает файл с указанным содержимым.

    path        — путь к файлу относительно корня проекта.
                  Запись за пределы проекта ЗАПРЕЩЕНА.
    content     — текстовое содержимое файла.
    create_dirs — если True, создаёт промежуточные каталоги (по умолчанию: True).
    Возвращает подтверждение с размером и количеством строк.
    """
    resolved = _resolve_safe(path)
    if resolved is None:
        return f"Ошибка: путь '{path}' выходит за пределы проекта. Запись запрещена."

    # Запрещаем перезаписывать критические файлы
    _PROTECTED = {"pyproject.toml", ".env", "setup.py", "setup.cfg"}
    if resolved.name in _PROTECTED:
        return f"Ошибка: файл '{resolved.name}' защищён от перезаписи."

    if create_dirs:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    elif not resolved.parent.exists():
        return f"Ошибка: каталог '{resolved.parent}' не существует. Передайте create_dirs=True."

    existed = resolved.exists()
    try:
        resolved.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Ошибка записи файла '{path}': {e}"

    size_bytes = resolved.stat().st_size
    size_label = f"{size_bytes} байт" if size_bytes < 1024 else f"{size_bytes / 1024:.1f} КБ"
    rel = resolved.relative_to(_PROJECT_ROOT)
    action = "обновлён" if existed else "создан"
    return (
        f"Файл {action} успешно!\n"
        f"  Путь:   {rel}\n"
        f"  Размер: {size_label}\n"
        f"  Строк:  {content.count(chr(10)) + 1}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# get_file_diff
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_file_diff(path: str, new_content: str) -> str:
    """Показывает unified diff между текущим содержимым файла и новым.

    path        — путь к файлу относительно корня проекта.
    new_content — новое содержимое, которое планируется записать.
    Если файл не существует — diff от пустого файла к новому содержимому.
    """
    resolved = _resolve_safe(path)
    if resolved is None:
        return f"Ошибка: путь '{path}' выходит за пределы проекта."

    if resolved.exists() and resolved.is_file():
        try:
            old_text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Ошибка чтения '{path}': {e}"
        old_lines = old_text.splitlines(keepends=True)
        label_old = str(resolved.relative_to(_PROJECT_ROOT))
    else:
        old_lines = []
        label_old = "/dev/null"

    new_lines = new_content.splitlines(keepends=True)
    rel = path if not os.path.isabs(path) else str(Path(path).relative_to(_PROJECT_ROOT))

    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{label_old}",
        tofile=f"b/{rel}",
        lineterm="",
    ))

    if not diff:
        return "Различий нет — содержимое идентично текущему файлу."

    lines_added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    lines_removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    header = f"Diff для '{rel}': +{lines_added} строк, -{lines_removed} строк\n"
    return header + "\n".join(diff)


# ─────────────────────────────────────────────────────────────────────────────
# get_project_structure
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_project_structure(directory: str = ".", depth: int = 3) -> str:
    """Возвращает дерево каталогов и файлов проекта.

    directory — корневой каталог для дерева (по умолчанию: '.', весь проект).
    depth     — максимальная глубина обхода (по умолчанию: 3).
    Системные каталоги (.git, __pycache__, .venv и т.д.) пропускаются.
    """
    resolved_dir = _resolve_safe(directory)
    if resolved_dir is None:
        return f"Ошибка: каталог '{directory}' выходит за пределы проекта."
    if not resolved_dir.exists():
        return f"Ошибка: каталог '{directory}' не существует."
    if not resolved_dir.is_dir():
        return f"Ошибка: '{directory}' — не каталог."

    lines: list[str] = []
    try:
        rel_root = resolved_dir.relative_to(_PROJECT_ROOT)
        lines.append(f"{rel_root}/")
    except ValueError:
        lines.append(f"{resolved_dir.name}/")

    def _walk(current: Path, prefix: str, current_depth: int) -> None:
        if current_depth > depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and e.name not in _SKIP_DIRS and not e.name.startswith(".")]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]

        all_visible = dirs + files
        for i, entry in enumerate(all_visible):
            is_last = i == len(all_visible) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                _walk(entry, prefix + extension, current_depth + 1)

    _walk(resolved_dir, "", 1)

    total_lines = len(lines)
    if total_lines > 200:
        lines = lines[:200]
        lines.append(f"... (показано 200 из {total_lines} элементов, увеличьте depth или уточните directory)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# get_git_log
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_git_log(n: int = 10, oneline: bool = False) -> str:
    """Возвращает последние N коммитов из git log.

    n       — количество коммитов (по умолчанию: 10).
    oneline — если True, выводит краткий формат (хеш + тема); иначе полный формат.
    """
    fmt = "--oneline" if oneline else "--format=%H%n%an%n%ad%n%s%n%b%n---"
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_PROJECT_ROOT), "log", f"-{n}", fmt],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return "История коммитов пуста."
        return out
    except subprocess.CalledProcessError:
        return "Ошибка: не удалось выполнить git log (не git-репозиторий?)."
    except FileNotFoundError:
        return "Ошибка: git не установлен."


if __name__ == "__main__":
    mcp.run()
